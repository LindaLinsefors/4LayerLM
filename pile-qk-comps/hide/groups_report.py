"""Per-matrix component-group reports: L<l>_<q|k>.md + pairwise grid plots.

For each module in groups_def.GROUPS, writes pile-qk-comps/L<l>/L<l>_<q|k>.md
(figures in pile-qk-comps/hide/figures/L<l>-Attn-<q|k>/) listing every group (members, autointerp labels, mean CI,
CI-firing count in the sample) and, per group with >1 member, one PNG with
four annotated pairwise grids:
  co-activation  : Pearson r of |a_c| over the 1.02M-token sample
  co-CI          : Pearson r of causal importance over the same sample
  input cosine   : |cos(V_a, V_b)| of the read-in vectors
  output cosine  : |cos(U_a, U_b)| of the write vectors
Stats from cache/groups_stats.npz (see groups_compute.py); V, U from energies.npz.

Usage: python groups_report.py [module ...]   (default: all in GROUPS)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT_BASE = HERE.parent
sys.path.insert(0, str(HERE))
from groups_def import GROUPS

SITE = json.load(open(HERE / "cache" / "site_descriptions.json"))

STATS_PATH = HERE / "cache" / "groups_stats.npz"
STATS = np.load(STATS_PATH) if STATS_PATH.exists() else None   # None: text-only reports
ENERGIES = np.load(HERE / "cache" / "energies.npz")
MEAN_CI = np.load(HERE / "cache" / "mean_ci.npz")
T = int(STATS["T"]) if STATS is not None else 0
INTERP_DB = (ROOT / "prev_paper" / "models" / "pile_4layer"
             / "additional-component-data" / "interp.db")


def folder_name(mod: str) -> str:
    l, kind = mod.split(".")[1], mod.split(".")[3][0]
    return f"L{l}-Attn-{kind}"


def report_stem(mod: str) -> str:
    l, kind = mod.split(".")[1], mod.split(".")[3][0]
    return f"L{l}_{kind}"


def pearson_from_stats(S1: np.ndarray, G: np.ndarray) -> np.ndarray:
    m = S1 / T
    cov = G / T - np.outer(m, m)
    s = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.outer(s, s)
    return r


def grid_panel(ax, M, ids, title, cmap, vmin, vmax):
    disp = np.ma.masked_invalid(M)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(disp, cmap=cm, vmin=vmin, vmax=vmax)
    n = len(ids)
    fs = max(5, min(9, 90 // n))
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                dark = abs(M[i, j]) > 0.6 * max(abs(vmin), abs(vmax))
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=fs, color="white" if dark else "#222222")
    ax.set_xticks(range(n), ids, rotation=90, fontsize=fs + 1)
    ax.set_yticks(range(n), ids, fontsize=fs + 1)
    ax.set_title(title, fontsize=10)
    return im


def make_module(mod: str) -> None:
    out_dir = HERE / "figures" / folder_name(mod)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.where(MEAN_CI[mod] > 1e-6)[0]
    pos = {int(c): k for k, c in enumerate(idx)}
    if STATS is not None:
        assert (STATS[f"{mod}|idx"] == idx).all()
        r_act = pearson_from_stats(STATS[f"{mod}|S1a"], STATS[f"{mod}|Ga"])
        r_ci = pearson_from_stats(STATS[f"{mod}|S1c"], STATS[f"{mod}|Gc"])
        fires = STATS[f"{mod}|F"]
    V = ENERGIES[f"{mod}|V"]
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos = np.abs(Vn @ Vn.T)
    U = ENERGIES[f"{mod}|U"]
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    cos_u = np.abs(Un @ Un.T)
    mci = MEAN_CI[mod]

    site_key = mod.replace("h.", "").replace("attn.", "attn.").replace("_proj", "")

    def label(c: int) -> str:
        return SITE[f"{site_key}:{c}"]["label"]

    grouped = [c for _, _, ms in GROUPS[mod] for c in ms]
    assert len(grouped) == len(set(grouped)), f"duplicate members in {mod}"
    assert set(grouped) <= set(int(c) for c in idx), f"non-alive members in {mod}"
    ungrouped = sorted(set(int(c) for c in idx) - set(grouped))
    groups = [(s, t, sorted(ms)) for s, t, ms in GROUPS[mod]]
    groups.append(("ungrouped", "Ungrouped", ungrouped))
    fig_dir = f"../hide/figures/{folder_name(mod)}"
    lines = [
        f"# {report_stem(mod)} — component groups by expected CI co-firing ({mod})",
        "",
        f"All {len(idx)} alive components (harvest mean CI > 1e-6). Groups were "
        "formed by Claude (2026-08-28) reading each component's **full website "
        "description** (label + autointerp reasoning, fetched from the paper "
        "site) and grouping components expected to have high per-token "
        "CI-similarity: same trigger tokens/positions, subset-detectors of one "
        "pattern merged; patterns on different tokens kept apart; bimodal or "
        "vague descriptions left ungrouped. No activation/CI data was used to "
        "form the groups.",
        "",
        "Per group with >1 member, four pairwise grids over a "
        + (f"{T:,}-token Pile sample (4000 cached rows):" if STATS is not None
           else "large Pile sample (figures pending — stats computing on Modal):"),
        "",
        "- **co-activation** — Pearson r of |a_c| = |‖U_c‖ (V_c·φ)| per token",
        "- **co-CI** — Pearson r of the causal importance (lower_leaky) per token;"
        " gray = component never varied (no CI in sample)",
        "- **input cosine** — |cos(V_a, V_b)| of read-in vectors"
        " (|·|: component sign is gauge)",
        "- **output cosine** — |cos(U_a, U_b)| of write vectors"
        " (|·|: same gauge)",
        "",
        "'fires' below = tokens with CI > 0.1 in the sample. Within each "
        "group, components are ordered by component id (in the lists and on "
        "all grid axes).",
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

    for slug, title, members in groups:
        lines += [f'<a id="{slug}"></a>', "", f"### {title} ({len(members)})", ""]
        for c in members:
            fire_str = f", fires {int(fires[pos[c]])}" if STATS is not None else ""
            lines.append(f"- **{c}** (mean CI {mci[c]:.4f}{fire_str}): {label(c)}")
        lines.append("")
        if len(members) < 2 or slug == "ungrouped":
            continue
        if STATS is None:
            lines += [f"![{title}]({fig_dir}/{slug}.png)", ""]
            continue
        sel = [pos[c] for c in members]
        fig_w = max(12, 0.55 * len(members) * 4 + 4)
        fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_w / 4 + 0.8),
                                 constrained_layout=True)
        grid_panel(axes[0], r_act[np.ix_(sel, sel)], members,
                   "co-activation  r(|a|)", "RdBu_r", -1, 1)
        grid_panel(axes[1], r_ci[np.ix_(sel, sel)], members,
                   "co-CI  r(CI)", "RdBu_r", -1, 1)
        grid_panel(axes[2], cos[np.ix_(sel, sel)], members,
                   "input |cos(V)|", "Blues", 0, 1)
        grid_panel(axes[3], cos_u[np.ix_(sel, sel)], members,
                   "output |cos(U)|", "Blues", 0, 1)
        fig.suptitle(f"{mod} — {title}", fontsize=11)
        fig.savefig(out_dir / f"{slug}.png", dpi=150)
        plt.close(fig)
        lines += [f"![{title}]({fig_dir}/{slug}.png)", ""]

    if STATS is not None:
        order = [pos[c] for _, _, ms in groups for c in ms]
        bounds = np.cumsum([len(ms) for _, _, ms in groups])
        n = len(order)
        fig, axes = plt.subplots(1, 4, figsize=(25, 6.9), constrained_layout=True)
        panels = [(r_act, "co-activation  r(|a|)", "RdBu_r", -1, 1),
                  (r_ci, "co-CI  r(CI)", "RdBu_r", -1, 1),
                  (cos, "input |cos(V)|", "Blues", 0, 1),
                  (cos_u, "output |cos(U)|", "Blues", 0, 1)]
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
        fig.suptitle(f"{mod} — all {n} alive components, ordered by group",
                     fontsize=12)
        fig.savefig(out_dir / "all_pairs.png", dpi=200)
        plt.close(fig)
        lines += ["## All pairs", "",
                  "All alive components in group order (black lines: group "
                  "boundaries; last block: ungrouped).", "",
                  f"![all pairs]({fig_dir}/all_pairs.png)", ""]

        # large single-panel co-CI grid with every component index labeled
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
                     "(group order)", fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.savefig(out_dir / "co_ci_full.png", dpi=150)
        plt.close(fig)
        lines += ["## Full co-CI grid", "",
                  "The co-CI panel alone, large, with every component index "
                  "on both axes (same group order as above).", "",
                  f"![full co-CI grid]({fig_dir}/co_ci_full.png)", ""]

    layer_dir = OUT_BASE / f"L{mod.split('.')[1]}"
    layer_dir.mkdir(exist_ok=True)
    report_path = layer_dir / f"{report_stem(mod)}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report_path,
          f"({sum(1 for _, _, m in groups if len(m) > 1)} figures)")


if __name__ == "__main__":
    mods = sys.argv[1:] or list(GROUPS)
    for mod in mods:
        make_module(mod)
