"""Second co-CI-clustered per-matrix report (alive components only), with
threshold rules instead of hierarchical linkage (user decision 2026-09-02):

  1. Two components with co-CI r(CI) > 0.9 belong to the same cluster
     (pairs processed in descending r, chains allowed).
  2. A component does not join a cluster if it has r(CI) <= 0.0 with any
     existing member (equally, two clusters only merge if every cross pair
     has r > 0.0); such joins are skipped and counted.

Clusters with >= 2 members become groups.  Groups are ordered by their
members' average mean CI (descending), members within a group by mean CI
(descending), Ungrouped likewise — unlike the linkage report, nothing is
ordered by size or component id.

Writes pile-qk-comps/L<l>/L<l>_<q|k>_coci09.md, figures in
pile-qk-comps/hide/figures/L<l>-Attn-<q|k>-coci09/. Same figure set as
groups_report.py (four pairwise grids incl. output |cos(U)|).

Usage: python groups_coci09_report.py [module ...]
       (default: h.0.attn.q_proj h.0.attn.k_proj)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from groups_report import (ENERGIES, MEAN_CI, OUT_BASE, SITE, STATS, T,
                           folder_name, grid_panel, pearson_from_stats,
                           report_stem)

R_EDGE, R_BLOCK = 0.9, 0.0


def cluster(Rf: np.ndarray) -> tuple[list[list[int]], int]:
    """Rule-based clusters over positions 0..n-1; returns (clusters, blocked)."""
    n = len(Rf)
    pairs = sorted(((Rf[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                    if Rf[i, j] > R_EDGE), reverse=True)
    of: dict[int, int] = {}          # position -> cluster id
    members: dict[int, set] = {}     # cluster id -> positions
    nxt = blocked = 0
    for _, i, j in pairs:
        a, b = of.get(i), of.get(j)
        if a is None and b is None:
            members[nxt] = {i, j}
            of[i] = of[j] = nxt
            nxt += 1
        elif a == b:
            continue
        elif a is None or b is None:
            k, new = (b, i) if a is None else (a, j)
            if all(Rf[new, m] > R_BLOCK for m in members[k]):
                members[k].add(new)
                of[new] = k
            else:
                blocked += 1
        else:
            if all(Rf[p, q] > R_BLOCK for p in members[a] for q in members[b]):
                for q in members[b]:
                    of[q] = a
                members[a] |= members.pop(b)
            else:
                blocked += 1
    return [sorted(ms) for ms in members.values()], blocked


def make_module(mod: str) -> None:
    out_dir = HERE / "figures" / f"{folder_name(mod)}-coci09"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.where(MEAN_CI[mod] > 1e-6)[0]
    pos = {int(c): k for k, c in enumerate(idx)}
    assert (STATS[f"{mod}|idx"] == idx).all()
    r_act = pearson_from_stats(STATS[f"{mod}|S1a"], STATS[f"{mod}|Ga"])
    r_ci = pearson_from_stats(STATS[f"{mod}|S1c"], STATS[f"{mod}|Gc"])
    fires = STATS[f"{mod}|F"]
    cosV, cosU = ({}, {})
    for key, store in (("V", cosV), ("U", cosU)):
        M = ENERGIES[f"{mod}|{key}"]
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        store["c"] = np.abs(Mn @ Mn.T)
    cosV, cosU = cosV["c"], cosU["c"]
    mci = MEAN_CI[mod]

    site_key = mod.replace("_proj", "").replace("h.", "")

    def label(c: int) -> str:
        return SITE[f"{site_key}:{c}"]["label"]

    Rf = np.nan_to_num(r_ci, nan=0.0)
    raw, blocked = cluster(Rf)
    clusters = [sorted((int(idx[k]) for k in ms), key=lambda c: -mci[c])
                for ms in raw]
    clusters.sort(key=lambda ms: -np.mean([mci[c] for c in ms]))
    groups = [(f"cluster-{i + 1}", f"Cluster {i + 1}", ms)
              for i, ms in enumerate(clusters)]
    ungrouped = sorted(set(int(c) for c in idx)
                       - {c for _, _, ms in groups for c in ms},
                       key=lambda c: -mci[c])
    groups.append(("ungrouped", "Ungrouped", ungrouped))

    lines = [
        f"# {report_stem(mod)} (co-CI > 0.9 clustering) — threshold-rule "
        f"component clusters ({mod})",
        "",
        f"Alternative to [{report_stem(mod)}_coci.md]({report_stem(mod)}_coci.md)"
        " (average-linkage) with threshold rules instead (alive components "
        "only, as before): **(1)** two components with co-CI r(CI) > "
        f"{R_EDGE} are in the same cluster (pairs processed in descending r, "
        "chains allowed); **(2)** a component does not join a cluster if it "
        f"has r(CI) ≤ {R_BLOCK} with any existing member (two clusters only "
        "merge if every cross pair is > 0); such joins are skipped"
        f" ({blocked} skipped here). r(CI) over the {T:,}-token Pile sample."
        " Clusters with ≥ 2 members become groups. **Ordering is by mean CI "
        "throughout**: groups by their members' average mean CI "
        "(descending), members within a group — and Ungrouped — by mean CI "
        "(descending).",
        "",
        "Per group with >1 member, four pairwise grids:",
        "",
        "- **co-activation** — Pearson r of |a_c| = |‖U_c‖ (V_c·φ)| per token",
        "- **co-CI** — Pearson r of the causal importance (lower_leaky) per"
        " token; gray = component never varied (no CI in sample)",
        "- **input cosine** — |cos(V_a, V_b)| of read-in vectors"
        " (|·|: component sign is gauge)",
        "- **output cosine** — |cos(U_a, U_b)| of write vectors (same gauge)",
        "",
        "'fires' below = tokens with CI > 0.1 in the sample. Grid axes use "
        "the same mean-CI order as the lists.",
        "",
        "## Groups",
        "",
        "| group | n | members |",
        "|---|---|---|",
    ]
    for slug, title, members in groups:
        lines.append(f"| [{title}](#{slug}) | {len(members)} | "
                     + " ".join(str(c) for c in members) + " |")
    lines.append("")

    fig_dir = f"../hide/figures/{out_dir.name}"
    for slug, title, members in groups:
        lines += [f'<a id="{slug}"></a>', "", f"### {title} ({len(members)})", ""]
        for c in members:
            lines.append(f"- **{c}** (mean CI {mci[c]:.2e}, "
                         f"fires {int(fires[pos[c]])}): {label(c)}")
        lines.append("")
        if len(members) < 2 or slug == "ungrouped":
            continue
        sel = [pos[c] for c in members]
        fig_w = max(12, 0.55 * len(members) * 4 + 5)
        fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_w / 4 + 0.8),
                                 constrained_layout=True)
        grid_panel(axes[0], r_act[np.ix_(sel, sel)], members,
                   "co-activation  r(|a|)", "RdBu_r", -1, 1)
        grid_panel(axes[1], r_ci[np.ix_(sel, sel)], members,
                   "co-CI  r(CI)", "RdBu_r", -1, 1)
        grid_panel(axes[2], cosV[np.ix_(sel, sel)], members,
                   "input |cos(V)|", "Blues", 0, 1)
        grid_panel(axes[3], cosU[np.ix_(sel, sel)], members,
                   "output |cos(U)|", "Blues", 0, 1)
        fig.suptitle(f"{mod} — {title}", fontsize=11)
        fig.savefig(out_dir / f"{slug}.png", dpi=150)
        plt.close(fig)
        lines += [f"![{title}]({fig_dir}/{slug}.png)", ""]

    order = [pos[c] for _, _, ms in groups for c in ms]
    bounds = np.cumsum([len(ms) for _, _, ms in groups])
    n = len(order)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.6), constrained_layout=True)
    panels = [(r_act, "co-activation  r(|a|)", "RdBu_r", -1, 1),
              (r_ci, "co-CI  r(CI)", "RdBu_r", -1, 1),
              (cosV, "input |cos(V)|", "Blues", 0, 1),
              (cosU, "output |cos(U)|", "Blues", 0, 1)]
    for ax, (M, title, cmap, vmin, vmax) in zip(axes, panels):
        disp = np.ma.masked_invalid(M[np.ix_(order, order)])
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#d9d9d9")
        im = ax.imshow(disp, cmap=cm, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color="#222222", lw=0.6)
            ax.axvline(b - 0.5, color="#222222", lw=0.6)
        centers = bounds - np.diff(np.concatenate(([0], bounds))) / 2
        ax.set_xticks(centers, [s for s, _, _ in groups], rotation=90,
                      fontsize=6)
        ax.set_yticks(centers, [s for s, _, _ in groups], fontsize=6)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"{mod} — all {n} alive components, cluster order "
                 "(clusters by avg mean CI)", fontsize=12)
    fig.savefig(out_dir / "all_pairs.png", dpi=200)
    plt.close(fig)
    lines += ["## All pairs", "",
              "All alive components in cluster order (black lines: cluster "
              "boundaries; last block: ungrouped).", "",
              f"![all pairs]({fig_dir}/all_pairs.png)", ""]

    ids = [c for _, _, ms in groups for c in ms]
    side = max(10.0, 0.14 * n + 2)
    fig, ax = plt.subplots(figsize=(side, side), constrained_layout=True)
    disp = np.ma.masked_invalid(r_ci[np.ix_(order, order)])
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(disp, cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="#222222", lw=0.6)
        ax.axvline(b - 0.5, color="#222222", lw=0.6)
    fs = max(5, min(10, int((side - 2) * 72 / n * 0.6)))
    ax.set_xticks(range(n), ids, rotation=90, fontsize=fs)
    ax.set_yticks(range(n), ids, fontsize=fs)
    ax.tick_params(length=0)
    ax.set_title(f"{mod} — co-CI  r(CI), all {n} alive components "
                 "(cluster order)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.savefig(out_dir / "co_ci_full.png", dpi=150)
    plt.close(fig)
    lines += ["## Full co-CI grid", "",
              "The co-CI panel alone, large, with every component index "
              "on both axes (same cluster order as above).", "",
              f"![full co-CI grid]({fig_dir}/co_ci_full.png)", ""]

    layer_dir = OUT_BASE / f"L{mod.split('.')[1]}"
    layer_dir.mkdir(exist_ok=True)
    report_path = layer_dir / f"{report_stem(mod)}_coci09.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report_path, f"({len(clusters)} clusters, "
          f"{len(ungrouped)} ungrouped, {blocked} joins blocked)")


if __name__ == "__main__":
    for mod in sys.argv[1:] or ["h.0.attn.q_proj", "h.0.attn.k_proj"]:
        make_module(mod)
