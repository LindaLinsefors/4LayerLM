"""Corpus-frequency views of the token embeddings, one column per model.

Rows:
  1. scatter: corpus count vs embedding-row norm (all tokens)
  2. histogram of corpus counts -- one bin for count 0, geometric bins from 1,
     drawn as equal-width bars over bin index (gaps = bins holding no integer);
     integer counts 0-13 and the alive cutoff marked
  3. scatter: corpus count vs projection onto PC1 of the alive-token PCA
     (all tokens; sign fixed so the alive mean is >= 0)

Frequency sources:
  pile_4l   -- token counts over the 4,000 cached Pile training rows
               (context-loss/cache/pile_rows.pt, 4000 x 513 ~= 2.05M tokens).
  simple_2l -- the `freq` column of simple-token-table/token_table.csv
               (counts over 20,000 stories, 5.74M tokens).
Cutoffs (CLAUDE.md 2026-08-27): count >= 7 (pile_4l "frequent"), >= 10
(simple_2l "alive"). Pile scatters are colored by the pile token categories
frequent / unfrequent / missing (missing = the 304 never-in-training tokens
from missing_tokens.npy); simple_2l keeps the old alive/dead terminology.

Output: token-embeds/freq.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent.parent / "freq.png"


pile_emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
simple_emb = torch.load(
    ROOT / "prev_paper/models/simplestories_2layer/target_model_gf6rbga0/model_step_99999.pt",
    map_location="cpu", weights_only=True,
)["wte.weight"].float().numpy()

pile_rows = torch.stack(torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
))
pile_counts = np.bincount(pile_rows.flatten().numpy(), minlength=len(pile_emb))

table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
simple_counts = np.zeros(len(simple_emb), dtype=np.int64)
simple_counts[table["id"].to_numpy()] = table["freq"].to_numpy()

panels = [
    ("pile_4l", "pile_4l  (counts over 2.05M cached Pile tokens)",
     pile_emb, pile_counts, 7),
    ("simple_2l", "simple_2l  (counts over 20k stories, 5.74M tokens)",
     simple_emb, simple_counts, 10),
]

DOT = BAR = "#4e79a7"
CUT = "#b3502d"
CAT_COLORS = {"frequent": "#4e79a7", "unfrequent": "#e0793d", "missing": "#a1443a"}

missing_ids = np.load(Path(__file__).parent / "missing_tokens.npy")

rng = np.random.default_rng(0)
fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), height_ratios=[3, 1.4, 3])
for (ax, hist_ax, pc_ax), (name, title, emb, counts, thr) in zip(axes.T, panels):
    norms = np.linalg.norm(emb, axis=1)
    alive = counts >= thr  # pile terminology: "frequent"

    if name == "pile_4l":
        missing = np.zeros(len(emb), dtype=bool)
        missing[missing_ids] = True
        categories = [("frequent", alive), ("unfrequent", ~alive & ~missing),
                      ("missing", missing)]
    else:
        categories = None  # simple_2l keeps single-hue scatters

    # jitter so integer-count columns (and the count-0 strip) read as density
    x = np.where(
        counts > 0,
        counts * np.exp(rng.uniform(-0.08, 0.08, len(counts))),
        rng.uniform(-0.35, 0.35, len(counts)),
    )
    if categories:
        for label, sel in categories:
            ax.scatter(x[sel], norms[sel], s=3, alpha=0.25, label=label,
                       color=CAT_COLORS[label], edgecolors="none", rasterized=True)
        ax.legend(fontsize=8, frameon=False, markerscale=3, loc="upper right")
    else:
        ax.scatter(x, norms, s=3, alpha=0.15, color=DOT, edgecolors="none",
                   rasterized=True)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("corpus token count (symlog; 0 = never seen)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    seen = counts > 0
    rho = spearmanr(counts[seen], norms[seen]).statistic
    ax.annotate(
        f"n = {len(counts):,}   never seen: {(~seen).sum():,}\n"
        f"Spearman ρ(count, norm | count>0) = {rho:+.2f}",
        xy=(0.02, 0.02), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=8.5, color="#333333",
    )

    # count histogram: one bin for count 0, log-spaced bins from 1, drawn as
    # equal-width bars over bin index (gaps = bins holding no integer)
    log_bins = np.logspace(0, np.log10(counts.max() + 1), 40)
    bin_edges = np.concatenate([[-0.5, 0.5], log_bins[1:]])
    heights, _ = np.histogram(counts, bins=bin_edges)
    hist_ax.bar(np.arange(len(heights)) + 0.5, heights, width=1.0,
                color=BAR, edgecolor="white", linewidth=0.3)

    def to_bar_x(v):  # count value -> bin-index coordinate (exact for v >= 0.5)
        return np.interp(np.log10(v), np.log10(bin_edges[1:]),
                         np.arange(1, len(bin_edges)))

    # mark where the integer counts 0..13 fall, at their position within the bin
    for n in range(14):
        b = np.searchsorted(bin_edges, n, side="right") - 1
        frac = np.clip((n - bin_edges[b]) / (bin_edges[b + 1] - bin_edges[b]), 0.2, 0.8)
        hist_ax.annotate(
            str(n), xy=(b + frac, heights[b]), xytext=(0, 2 + 6 * (n % 3)),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=7, color="#555555",
        )

    cut_label = "frequent" if name == "pile_4l" else "alive"
    cut_x = to_bar_x(thr - 0.5)
    hist_ax.axvline(cut_x, color=CUT, linestyle="--", linewidth=1.5)
    hist_ax.annotate(f"{cut_label} cutoff (count ≥ {thr})", xy=(cut_x, 0.95),
                     xycoords=("data", "axes fraction"), xytext=(4, 0),
                     textcoords="offset points", ha="left", va="top",
                     fontsize=8, color=CUT)

    tick_vals = [10 ** k for k in range(int(np.log10(counts.max())) + 1)]
    hist_ax.set_xticks([0.5, *to_bar_x(np.array(tick_vals))])
    hist_ax.set_xticklabels(["0", *(f"$10^{k}$" if k > 1 else str(10 ** k)
                                    for k in range(len(tick_vals)))])
    hist_ax.set_xlabel("corpus token count (equal-width bins)")
    hist_ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    hist_ax.spines[["top", "right"]].set_visible(False)

    # projection onto PC1 of the alive-token PCA, all tokens
    centered = emb[alive] - emb[alive].mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = emb @ vt[0]
    if pc1[alive].mean() < 0:
        pc1 = -pc1
    if categories:
        for label, sel in categories:
            pc_ax.scatter(x[sel], pc1[sel], s=3, alpha=0.25, label=label,
                          color=CAT_COLORS[label], edgecolors="none", rasterized=True)
        pc_ax.legend(fontsize=8, frameon=False, markerscale=3, loc="upper right")
    else:
        pc_ax.scatter(x, pc1, s=3, alpha=0.15, color=DOT, edgecolors="none",
                      rasterized=True)
    pc_ax.set_xscale("symlog", linthresh=1)
    pc_ax.set_xlabel("corpus token count (symlog; 0 = never seen)")
    pc_ax.grid(alpha=0.25, linewidth=0.5)
    pc_ax.spines[["top", "right"]].set_visible(False)

    print(f"{title}: never seen {(~seen).sum()}, spearman(norm) {rho:+.3f}, "
          f"spearman(PC1|count>0) {spearmanr(counts[seen], pc1[seen]).statistic:+.3f}")

axes[0, 0].set_ylabel("‖embedding row‖₂")
axes[1, 0].set_ylabel("tokens")
axes[2, 0].set_ylabel("projection onto alive-PC1\n(sign: alive mean ≥ 0)")

fig.suptitle("Corpus frequency vs token-embedding norm and PC1", y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
