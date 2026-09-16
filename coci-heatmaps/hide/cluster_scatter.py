"""Scatter of cluster size vs cluster-average mean CI, per model, for the
co-CI>0.9 threshold clusters (rule_clusters from report_clustered09) over ALL
components — alive and dead, singletons included.

x = cluster average of harvest mean CI (log scale; exactly-zero clusters on a
"0" strip at the left edge), y = cluster size (linear, small jitter so the
size-1 mass doesn't fully overplot). Dashed green line at mean CI = 1e-6 =
the alive/dead cutoff. Color hue = layer; shade + marker = attn (dark circle)
vs MLP (light square).

Writes report_cluster_scatter.md + hide/figures/cluster_scatter_<model>.png.

Usage: python coci-heatmaps/hide/cluster_scatter.py   (~30 s)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent / "old"  # reports moved into old/ 2026-09-02 (old vs new A/B decompositions)
CACHE = HERE / "cache"
sys.path.insert(0, str(HERE))
from report import MATS, MODELS, N_LAYERS, TOKENS_DESC  # noqa: E402
from report_clustered09 import R_BLOCK, R_EDGE, rule_clusters  # noqa: E402

ALIVE_CUT, CUT_COLOR = 1e-6, "#008300"
LAYER_HUE = ["#2a78d6", "#1baf7a", "#4a3aa7", "#eb6834"]  # blue aqua violet orange

# One-line summaries of the interp.db member labels for every pile cluster
# with >= 20 members, written by Claude 2026-09-02 (keyed by matrix + smallest
# member id; regenerate the label dump with the snippet in the report text).
SUMMARIES = {
    ("h.1.attn.v_proj", 2): "Document boundaries: detects end-of-text/"
        "section starts, suppresses cross-document continuation, initiates "
        "new-document content (the L1 EOS machinery).",
    ("h.1.mlp.c_fc", 63): "Structural-boundary / punctuation detectors that "
        "suppress continuation at delimiters and boundaries (includes 600, "
        "743, 2807 of the pos-0 massive-vector trigger set).",
    ("h.1.mlp.c_fc", 232): "Academic figure/table references: detects and "
        "produces 'Table N'/'Fig. N' labels and citations.",
    ("h.3.attn.v_proj", 5): "Dead (0/38 alive): incoherent low-confidence "
        "rare-token/punctuation labels — near-copy dead components, no "
        "coherent function.",
    ("h.1.mlp.c_fc", 71): "Dead (0/32 alive): incoherent low-confidence "
        "rare-punctuation/structural labels.",
    ("h.0.mlp.down_proj", 48): "Document/section boundaries: end-of-text -> "
        "new-section/topic starts, suppresses continuation (MLP-1 boundary "
        "machinery).",
    ("h.2.attn.k_proj", 23): "Structural/document boundary keys — the L2 "
        "attention-sink key block (first-token seq-start + EOS boundary-both "
        "groups merged).",
    ("h.1.attn.k_proj", 0): "Document-boundary keys: end-of-text -> new "
        "document/topic transition (the L1k EOS block).",
    ("h.0.attn.q_proj", 1): "The hacker-news title/username hyphen group "
        "(per the site descriptions); local labels are vaguer — hyphens, "
        "separators, section/metadata boundaries.",
    ("h.1.attn.o_proj", 27): "Dead (0/26 alive): sparse/rare-token noise "
        "labels, mostly low confidence.",
    ("h.1.mlp.down_proj", 43): "Dead (0/26 alive): punctuation/rare-token "
        "noise labels, mostly low confidence.",
    ("h.0.mlp.c_fc", 272): "Articles & determiners: detects 'the'/'a' and "
        "common function words, mostly suppressing them.",
    ("h.1.mlp.down_proj", 9): "Academic figure/table references (write side; "
        "pairs with the c_fc table/figure cluster).",
    ("h.2.attn.v_proj", 112): "Dead (0/20 alive): scattered punctuation/"
        "structure noise labels, mostly low confidence.",
    ("h.2.attn.v_proj", 3): "Academic figure/table references again at L2 "
        "(value side): detects/suppresses 'Table N'/'Fig. N' reference "
        "markers.",
    ("h.2.attn.o_proj", 14): "Dead (0/20 alive): sparse structural/"
        "punctuation noise labels, mostly low confidence.",
}
TABLE_MIN = 20


def shade(hex_color: str, light: bool) -> str:
    if not light:
        return hex_color
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{r + (255 - r) // 2:02x}{g + (255 - g) // 2:02x}{b + (255 - b) // 2:02x}"


def main() -> None:
    rng = np.random.default_rng(0)
    zero_strip = {}
    lines = [
        "# Cluster size vs average CI — co-CI>0.9 clusters, all components",
        "",
        "Each point is one cluster from the threshold rules of "
        "[report_all_clustered09.md](report_all_clustered09.md) (co-CI r > "
        f"{R_EDGE} chains, joins blocked at r ≤ {R_BLOCK}) over **all** "
        "components of every matrix — dead and alive, singletons included "
        "(size 1; y is jittered ±0.2 so they don't fully overplot). "
        "x = mean over members of the harvest mean CI, log scale; clusters "
        "whose average is exactly 0 sit on the '0' strip at the left edge. "
        "The dashed green line is the alive/dead cutoff (mean CI = 1e-6). "
        "Hue = layer; dark circles = attention matrices, light squares = MLP "
        "matrices.",
        "",
    ]
    for model in MODELS:
        stats = np.load(CACHE / f"coci_{model}.npz")
        mci = np.load(CACHE / f"mean_ci_{model}.npz")
        pts = []                       # (avg, size, layer, is_mlp)
        big_clusters = []              # (size, mod, min member, avg, n alive)
        for l in range(N_LAYERS[model]):
            for mat in MATS:
                mod = f"h.{l}.{mat}"
                m = mci[mod]
                order = np.argsort(-m, kind="stable")
                multi, _ = rule_clusters(stats[f"{mod}|r"].astype(np.float32),
                                         order)
                in_multi = {int(c) for cl in multi for c in cl}
                is_mlp = mat.startswith("mlp")
                for cl in multi:
                    pts.append((m[cl].mean(), len(cl), l, is_mlp))
                    if len(cl) >= TABLE_MIN:
                        big_clusters.append((len(cl), mod, int(cl.min()),
                                             m[cl].mean(),
                                             int((m[cl] > 1e-6).sum())))
                pts += [(m[c], 1, l, is_mlp) for c in order
                        if int(c) not in in_multi]

        avg = np.array([p[0] for p in pts])
        size = np.array([p[1] for p in pts])
        layer = np.array([p[2] for p in pts])
        mlp = np.array([p[3] for p in pts])
        floor = 10 ** np.floor(np.log10(avg[avg > 0].min()) - 1)
        x = np.where(avg > 0, avg, floor)

        for logy in (False, True) if model == "pile_4l" else (False,):
            # additive jitter on linear y, multiplicative on log y
            y = (size * np.exp(rng.uniform(-0.08, 0.08, len(size))) if logy
                 else size + rng.uniform(-0.2, 0.2, len(size)))
            fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            for l in range(N_LAYERS[model]):
                for is_mlp in (False, True):
                    sel = (layer == l) & (mlp == is_mlp)
                    single = sel & (size == 1)
                    big = sel & (size > 1)
                    kw = dict(color=shade(LAYER_HUE[l], is_mlp),
                              marker="s" if is_mlp else "o",
                              linewidths=0)
                    ax.scatter(x[single], y[single], s=8, alpha=0.25, **kw)
                    ax.scatter(x[big], y[big], s=30, alpha=0.9,
                               label=f"L{l} {'MLP' if is_mlp else 'attn'}",
                               **kw)
            ax.axvline(ALIVE_CUT, color=CUT_COLOR, lw=1.2, ls="--")
            ax.text(ALIVE_CUT, ax.get_ylim()[1], "dead | alive ",
                    color=CUT_COLOR, fontsize=9, ha="right", va="top")
            ticks = [10.0 ** e for e in
                     range(int(np.log10(floor)), 1, 3)]
            ax.set_xticks([floor] + ticks[1:])
            ax.set_xticklabels(["0"]
                               + [f"1e{int(np.log10(t))}" for t in ticks[1:]])
            if logy:
                yt = [t for t in (1, 2, 3, 5, 10, 20, 30, 50, 70)
                      if t <= 2 * size.max()]
                ax.set_yticks(yt)
                ax.set_yticklabels([str(t) for t in yt])
                ax.minorticks_off()
            ax.set_xlabel("cluster average of harvest mean CI"
                          " (log; exact zeros on the '0' strip)")
            ax.set_ylabel("cluster size" + (" (log)" if logy else ""))
            ax.set_title(f"{model} — co-CI>0.9 clusters: size vs average CI\n"
                         f"({len(pts):,} clusters, "
                         f"{int((size > 1).sum())} multi-member; legend "
                         "markers = multi-member, faint = singletons)",
                         fontsize=10)
            ax.legend(fontsize=8, ncols=2)
            suffix = "_logy" if logy else ""
            fig_path = HERE / "figures" / f"cluster_scatter_{model}{suffix}.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
        zero_strip[model] = int((avg == 0).sum())

        lines += [
            f"## {model}",
            "",
            f"{len(pts):,} clusters ({int((size > 1).sum())} with ≥ 2 members,"
            f" largest {int(size.max())}); {zero_strip[model]} zero-CI"
            f" clusters on the '0' strip. CI from {TOKENS_DESC[model]}.",
            "",
            f"![{model}](../hide/figures/cluster_scatter_{model}.png)",
            "",
        ]
        if model == "pile_4l":
            lines += ["Same plot, log y:", "",
                      f"![{model} log y]"
                      f"(hide/figures/cluster_scatter_{model}_logy.png)",
                      ""]
            # stacked histogram of per-COMPONENT mean CI, same color coding
            groups = [(l, is_mlp) for l in range(N_LAYERS[model])
                      for is_mlp in (False, True)]
            vals = {g: np.concatenate(
                        [mci[f"h.{g[0]}.{mat}"] for mat in MATS
                         if mat.startswith("mlp") == g[1]]) for g in groups}
            allv = np.concatenate(list(vals.values()))
            lo = 10 ** np.floor(np.log10(allv[allv > 0].min()))
            edges = np.logspace(np.log10(lo), 0, int(3 * np.log10(1 / lo)) + 1)
            zw = edges[1] / edges[0]                    # one-bin width ratio
            z_left = lo / zw ** 2                       # zero strip position
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
            ax.set_xscale("log")
            bottom = np.zeros(len(edges) - 1)
            zbot = 0.0
            for g in groups:
                l, is_mlp = g
                h, _ = np.histogram(vals[g][vals[g] > 0], bins=edges)
                nz = int((vals[g] == 0).sum())
                c = shade(LAYER_HUE[l], is_mlp)
                lab = f"L{l} {'MLP' if is_mlp else 'attn'}"
                ax.bar(edges[:-1], h, width=np.diff(edges), align="edge",
                       bottom=bottom, color=c, label=lab, linewidth=0)
                ax.bar([z_left], [nz], width=z_left * (zw - 1), align="edge",
                       bottom=[zbot], color=c, linewidth=0)
                bottom += h
                zbot += nz
            ax.axvline(ALIVE_CUT, color=CUT_COLOR, lw=1.2, ls="--")
            ax.text(ALIVE_CUT, ax.get_ylim()[1], "dead | alive ",
                    color=CUT_COLOR, fontsize=9, ha="right", va="top")
            ticks = [10.0 ** e for e in range(int(np.log10(z_left)), 1, 3)]
            ax.set_xticks([z_left * zw ** 0.5] + ticks[1:])
            ax.set_xticklabels(["0"] + [f"1e{int(np.log10(t))}"
                                        for t in ticks[1:]])
            ax.set_xlabel("harvest mean CI (log; exact zeros in the '0' bar)")
            ax.set_ylabel("components per bin (stacked)")
            ax.set_title(f"{model} — mean CI of all {len(allv):,} components",
                         fontsize=10)
            ax.legend(fontsize=8, ncols=2)
            fig.savefig(HERE / "figures" / f"mean_ci_hist_{model}.png",
                        dpi=200)
            plt.close(fig)
            lines += [
                "### Mean-CI histogram, all pile components",
                "",
                f"Per-component (not per-cluster) harvest mean CI, all "
                f"{len(allv):,} components, stacked by layer and attn/MLP "
                "(same colors as the scatters; exact zeros in the '0' bar).",
                "",
                f"![mean CI histogram](../hide/figures/mean_ci_hist_{model}.png)",
                "",
            ]

            alive_big = sorted((t for t in big_clusters if t[4] > 0),
                               key=lambda t: -t[0])
            lines += [
                f"### The {len(alive_big)} biggest alive clusters"
                f" (size ≥ {TABLE_MIN}; dead clusters omitted)",
                "",
                "Function summarized (by Claude, 2026-09-02) from the "
                "members' local autointerp labels (`interp.db`). The omitted "
                "dead clusters (6 of them, sizes 20-38, avg CI ~1e-8) all "
                "carry noise-level low-confidence labels — autointerp saw "
                "only residual firings.",
                "",
                "| matrix | size | avg mean CI | alive | what the members'"
                " autointerp says |",
                "|---|---|---|---|---|",
            ]
            for n, mod, first, a, nal in alive_big:
                s = SUMMARIES.get((mod, first),
                                  "(no summary — rerun the label dump)")
                lines.append(f"| {mod} | {n} | {a:.1e} | {nal}/{n} | {s} |")
            lines.append("")
        print(f"{model}: {len(pts):,} clusters plotted")

    (OUT / "report_cluster_scatter.md").write_text("\n".join(lines),
                                                   encoding="utf-8")
    print("wrote", OUT / "report_cluster_scatter.md")


if __name__ == "__main__":
    main()
