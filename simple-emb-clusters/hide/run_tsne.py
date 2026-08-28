"""t-SNE of the de-frequencied alive-token embeddings, both removal variants.

Per variant (pc1 / vcov): four runs (perplexity 5/30/100 euclidean + 30
cosine), cached to cache/<variant>/tsne_*.npy. Figures per variant:
  figures/<variant>/tsne_grid.png       -- all four runs, colored by class
  figures/<variant>/tsne_annotated.png  -- main run, class colors + labels
  figures/<variant>/tsne_freq_norm.png  -- main run, colored by log-frequency
                                           and by residual row norm
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from common import (CLASS_COLORS, CLASS_NAMES, INK, INK_2, MUTED, VARIANTS,
                    cache_dir, new_fig, processed, save_fig, seq_cmap, style_ax)

RUNS = {
    "perp5": dict(perplexity=5, metric="euclidean"),
    "perp30": dict(perplexity=30, metric="euclidean"),
    "perp100": dict(perplexity=100, metric="euclidean"),
    "cos30": dict(perplexity=30, metric="cosine"),
}
MAIN = "cos30"  # the run used for the annotated / colored single plots


def tsne_coords(emb: np.ndarray, name: str, variant: str) -> np.ndarray:
    f = cache_dir(variant) / f"tsne_{name}.npy"
    if f.exists():
        return np.load(f)
    print(f"running t-SNE {variant}/{name} ...", flush=True)
    xy = TSNE(n_components=2, init="pca", learning_rate="auto",
              random_state=0, **RUNS[name]).fit_transform(emb)
    np.save(f, xy)
    return xy


def scatter_classes(ax, xy, classes, s=3.0):
    for c, (cname, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        m = classes == c
        if not m.any():  # e.g. "Capitalized"/"special": absent from alive set
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=s, c=color, linewidths=0,
                   alpha=0.85, label=f"{cname} ({m.sum()})")


def run_variant(variant: str):
    d = processed(variant)
    coords = {name: tsne_coords(d.X, name, variant) for name in RUNS}

    # --- 2x2 grid: the four runs, class-colored ------------------------------
    fig = new_fig(figsize=(9.5, 9.5))
    titles = {"perp5": "perplexity 5, euclidean",
              "perp30": "perplexity 30, euclidean",
              "perp100": "perplexity 100, euclidean",
              "cos30": "perplexity 30, cosine"}
    for i, name in enumerate(RUNS):
        ax = fig.add_subplot(2, 2, i + 1)
        style_ax(ax)
        scatter_classes(ax, coords[name], d.classes)
        ax.set_title(titles[name], fontsize=9, color=INK_2)
        ax.set_xticklabels([]); ax.set_yticklabels([])
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
                     frameon=False, markerscale=4, bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(f"t-SNE, alive tokens (n={len(d.tokens)}), centered, "
                 f"{variant} removed", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    save_fig(fig, "tsne_grid.png", variant)
    plt.close(fig)

    # --- main run, annotated -------------------------------------------------
    xy = coords[MAIN]
    fig = new_fig(figsize=(13, 13))
    ax = fig.add_subplot(111)
    style_ax(ax)
    scatter_classes(ax, xy, d.classes, s=5)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=False, markerscale=4)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # Label high-frequency tokens, greedily skipping labels that would collide.
    span = xy.max(0) - xy.min(0)
    min_dist = 0.022 * span.max()
    placed: list[np.ndarray] = []
    n_labels = 0
    for i in np.argsort(-d.freq)[:400]:
        if placed and np.min(np.linalg.norm(np.array(placed) - xy[i], axis=1)) < min_dist:
            continue
        placed.append(xy[i])
        ax.annotate(d.tokens[i], xy[i], fontsize=6, color=INK,
                    xytext=(2, 2), textcoords="offset points")
        n_labels += 1
    print(f"placed {n_labels} token labels")
    ax.set_title(f"t-SNE ({titles[MAIN]}), {variant} removed — labels: most "
                 f"frequent tokens", fontsize=12, color=INK)
    save_fig(fig, "tsne_annotated.png", variant)
    plt.close(fig)

    # --- main run colored by log-frequency and by residual norm --------------
    fig = new_fig(figsize=(12.5, 6))
    for j, (values, label) in enumerate([
        (np.log10(d.freq), "log10(corpus count)"),
        (np.linalg.norm(d.X, axis=1), "residual row norm (after removal)"),
    ]):
        ax = fig.add_subplot(1, 2, j + 1)
        style_ax(ax)
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=4, c=values, cmap=seq_cmap(),
                        linewidths=0, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label(label, fontsize=8, color=INK_2)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_visible(False)
        ax.set_title(label, fontsize=10, color=INK_2)
        ax.set_xticklabels([]); ax.set_yticklabels([])
    fig.suptitle(f"t-SNE ({titles[MAIN]}), {variant} removed",
                 fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "tsne_freq_norm.png", variant)
    plt.close(fig)


if __name__ == "__main__":
    for variant in VARIANTS:
        run_variant(variant)
