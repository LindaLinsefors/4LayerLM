"""Alternative per-matrix report: groups derived from the co-CI data itself.

Instead of the description-based groups in groups_def.py, components are
clustered on the measured co-CI matrix: average-linkage hierarchical
clustering on distance 1 - r(CI) (NaN r -> 0), cut at distance 0.5, i.e.
clusters keep an average within-cluster r(CI) >= 0.5. Clusters with >= 2
members become groups (largest first); singletons go to Ungrouped.

Writes pile-qk-comps/L<l>/L<l>_<q|k>_coci.md, figures in
pile-qk-comps/hide/figures/L<l>-Attn-<q|k>-coci/. Same figure set as
groups_report.py.

Usage: python groups_coci_report.py [module ...]   (default: h.0.attn.k_proj)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from groups_report import (ENERGIES, MEAN_CI, OUT_BASE, SITE, STATS, T,
                           folder_name, grid_panel, pearson_from_stats,
                           report_stem)

R_CUT = 0.5   # keep clusters with average within-cluster r(CI) >= this


def make_module(mod: str) -> None:
    out_dir = HERE / "figures" / f"{folder_name(mod)}-coci"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.where(MEAN_CI[mod] > 1e-6)[0]
    pos = {int(c): k for k, c in enumerate(idx)}
    assert (STATS[f"{mod}|idx"] == idx).all()
    r_act = pearson_from_stats(STATS[f"{mod}|S1a"], STATS[f"{mod}|Ga"])
    r_ci = pearson_from_stats(STATS[f"{mod}|S1c"], STATS[f"{mod}|Gc"])
    fires = STATS[f"{mod}|F"]
    V = ENERGIES[f"{mod}|V"]
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos = np.abs(Vn @ Vn.T)
    mci = MEAN_CI[mod]

    site_key = mod.replace("_proj", "")

    def label(c: int) -> str:
        return SITE[f"{site_key.replace('h.', '')}:{c}"]["label"]

    # cluster on co-CI: average linkage on 1 - r, cut at 1 - R_CUT
    Rf = np.nan_to_num(r_ci, nan=0.0)
    D = squareform(np.clip(1 - Rf, 0, None), checks=False)
    cl = fcluster(linkage(D, method="average"), t=1 - R_CUT,
                  criterion="distance")
    clusters = [sorted(int(idx[k]) for k in np.where(cl == i)[0])
                for i in np.unique(cl)]
    clusters = [ms for ms in clusters if len(ms) >= 2]
    clusters.sort(key=lambda ms: (-len(ms), ms[0]))
    groups = [(f"cluster-{i + 1}", f"Cluster {i + 1}", ms)
              for i, ms in enumerate(clusters)]
    ungrouped = sorted(set(int(c) for c in idx)
                       - {c for _, _, ms in groups for c in ms})
    groups.append(("ungrouped", "Ungrouped", ungrouped))

    lines = [
        f"# {report_stem(mod)} (co-CI clustering) — data-driven component "
        f"groups ({mod})",
        "",
        f"Alternative to [{report_stem(mod)}.md]({report_stem(mod)}.md): "
        "same alive components and figures, but the groups come from the "
        "**measured co-CI data**, not the website descriptions. "
        "Average-linkage hierarchical clustering on distance 1 − r(CI) over "
        f"the {T:,}-token Pile sample, cut at distance {1 - R_CUT} (so each "
        f"cluster keeps average within-cluster r(CI) ≥ {R_CUT}); clusters "
        "with ≥ 2 members become groups, largest first; singletons and "
        "never-varying components go to Ungrouped.",
        "",
        "Per group with >1 member, three pairwise grids:",
        "",
        "- **co-activation** — Pearson r of |a_c| = |‖U_c‖ (V_c·φ)| per token",
        "- **co-CI** — Pearson r of the causal importance (lower_leaky) per token;"
        " gray = component never varied (no CI in sample)",
        "- **input cosine** — |cos(V_a, V_b)| of read-in vectors"
        " (|·|: component sign is gauge)",
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

    fig_dir = f"../hide/figures/{out_dir.name}"
    for slug, title, members in groups:
        lines += [f'<a id="{slug}"></a>', "", f"### {title} ({len(members)})", ""]
        for c in members:
            lines.append(f"- **{c}** (mean CI {mci[c]:.4f}, "
                         f"fires {int(fires[pos[c]])}): {label(c)}")
        lines.append("")
        if len(members) < 2 or slug == "ungrouped":
            continue
        sel = [pos[c] for c in members]
        fig_w = max(9, 0.55 * len(members) * 3 + 4)
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_w / 3 + 0.8),
                                 constrained_layout=True)
        grid_panel(axes[0], r_act[np.ix_(sel, sel)], members,
                   "co-activation  r(|a|)", "RdBu_r", -1, 1)
        grid_panel(axes[1], r_ci[np.ix_(sel, sel)], members,
                   "co-CI  r(CI)", "RdBu_r", -1, 1)
        grid_panel(axes[2], cos[np.ix_(sel, sel)], members,
                   "input |cos(V)|", "Blues", 0, 1)
        fig.suptitle(f"{mod} — {title}", fontsize=11)
        fig.savefig(out_dir / f"{slug}.png", dpi=150)
        plt.close(fig)
        lines += [f"![{title}]({fig_dir}/{slug}.png)", ""]

    order = [pos[c] for _, _, ms in groups for c in ms]
    bounds = np.cumsum([len(ms) for _, _, ms in groups])
    n = len(order)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.9), constrained_layout=True)
    panels = [(r_act, "co-activation  r(|a|)", "RdBu_r", -1, 1),
              (r_ci, "co-CI  r(CI)", "RdBu_r", -1, 1),
              (cos, "input |cos(V)|", "Blues", 0, 1)]
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
    fig.suptitle(f"{mod} — all {n} alive components, ordered by co-CI cluster",
                 fontsize=12)
    fig.savefig(out_dir / "all_pairs.png", dpi=200)
    plt.close(fig)
    lines += ["## All pairs", "",
              "All alive components in cluster order (black lines: cluster "
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
    report_path = layer_dir / f"{report_stem(mod)}_coci.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report_path,
          f"({len(clusters)} clusters, {len(ungrouped)} ungrouped)")


if __name__ == "__main__":
    for mod in sys.argv[1:] or ["h.0.attn.k_proj"]:
        make_module(mod)
