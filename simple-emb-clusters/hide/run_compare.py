"""Statistics per variant + head-to-head comparison of the two removal variants.

Per variant (pc1 / vcov):
  - how much frequency signal survives: Spearman(freq, residual row norm),
    corr(log freq, residual norm), and CV-ridge decodability of log freq from
    the processed rows (the adversarial test from token-embeds analysis)
  - anisotropy: mean pairwise cosine before vs after processing
  - what the silhouette-optimal KMeans split separates (freq stats per side)

Cross-variant:
  - cos(v_pc1, v_vcov)
  - per-method ARI between the two variants' cluster labels
  - k-NN overlap: mean fraction of shared 10 nearest neighbors (cosine)
  - figures/compare_tsne.png: both t-SNE maps side by side (class + freq)

Writes cache/compare.json.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from common import (CACHE, INK, INK_2, MUTED, VARIANTS, cache_dir, new_fig,
                    processed, save_fig, seq_cmap, style_ax)
from run_tsne import MAIN, scatter_classes


def mean_pairwise_cos(Xn: np.ndarray) -> float:
    """mean_{i != j} cos(x_i, x_j), via |sum x|^2 = sum_ij cos."""
    s = Xn.sum(0).astype(np.float64)
    n = len(Xn)
    return float((s @ s - n) / (n * (n - 1)))


def cv_decodability(X: np.ndarray, y: np.ndarray, lam: float = 0.01) -> float:
    """CV Spearman of ridge-decoded log freq from the processed rows."""
    Xc, yc = X - X.mean(0), y - y.mean()
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(y)) % 5
    oof = np.empty(len(y))
    for k in range(5):
        tr = folds != k
        w = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(X.shape[1]),
                            Xc[tr].T @ yc[tr])
        oof[~tr] = Xc[~tr] @ w
    return float(spearmanr(oof, yc).statistic)


def main():
    data = {v: processed(v) for v in VARIANTS}
    out: dict = {"cos_v_pc1_v_vcov": float(data["pc1"].v @ data["vcov"].v)}

    # raw alive rows for the "before" anisotropy baseline + removed projections
    import common

    emb, all_tokens = common.embeddings_and_tokens()
    freq_all = common.frequencies(len(all_tokens))
    alive = freq_all >= common.ALIVE_THR
    A = emb[alive].astype(np.float64)
    An_raw = A / np.linalg.norm(A, axis=1, keepdims=True)
    Ac = A - A.mean(0)
    Acn = Ac / np.linalg.norm(Ac, axis=1, keepdims=True)
    y = np.log(freq_all[alive].astype(np.float64))
    out["anisotropy"] = {"raw alive rows": mean_pairwise_cos(An_raw),
                        "centered alive rows": mean_pairwise_cos(Acn)}

    for v in VARIANTS:
        d = data[v]
        norms = np.linalg.norm(d.X, axis=1)
        labels2 = np.load(cache_dir(v) / "labels.npz")["kmeans_opt"]
        k2 = {}
        for c in sorted(set(labels2)):
            m = labels2 == c
            k2[f"cluster {c}"] = dict(
                n=int(m.sum()), median_freq=float(np.median(d.freq[m])),
                suffix_frac=float(np.mean(d.classes[m] == 1)),
                examples=[d.tokens[i]
                          for i in np.where(m)[0][np.argsort(-d.freq[m])][:8]],
            )
        out[v] = dict(
            spearman_freq_residual_norm=float(spearmanr(d.freq, norms).statistic),
            corr_logfreq_residual_norm=float(np.corrcoef(np.log(d.freq), norms)[0, 1]),
            cv_freq_decodability=cv_decodability(d.X.astype(np.float64), y),
            anisotropy_processed=mean_pairwise_cos(d.Xn),
            spearman_freq_removed_proj=float(spearmanr(d.freq, Ac @ d.v).statistic),
            kmeans_opt_split=k2,
        )

    # --- per-method ARI between variants -------------------------------------
    la = np.load(cache_dir("pc1") / "labels.npz")
    lb = np.load(cache_dir("vcov") / "labels.npz")
    out["ari_pc1_vs_vcov"] = {k: float(adjusted_rand_score(la[k], lb[k]))
                              for k in la.files}

    # --- k-NN overlap ---------------------------------------------------------
    K = 10
    nn = {}
    for v in VARIANTS:
        S = data[v].Xn @ data[v].Xn.T
        np.fill_diagonal(S, -np.inf)
        nn[v] = np.argpartition(-S, K, axis=1)[:, :K]
    overlap = np.array([len(np.intersect1d(nn["pc1"][i], nn["vcov"][i])) / K
                        for i in range(len(nn["pc1"]))])
    out["knn10_overlap"] = dict(mean=float(overlap.mean()),
                                frac_ge_half=float(np.mean(overlap >= 0.5)))

    print(json.dumps(out, indent=1, ensure_ascii=False))
    (CACHE / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # --- side-by-side t-SNE figure -------------------------------------------
    fig = new_fig(figsize=(12.5, 12.5))
    for j, v in enumerate(VARIANTS):
        xy = np.load(cache_dir(v) / f"tsne_{MAIN}.npy")
        d = data[v]
        ax = fig.add_subplot(2, 2, j + 1)
        style_ax(ax)
        scatter_classes(ax, xy, d.classes)
        if j == 0:
            leg = ax.legend(loc="upper right", fontsize=7, frameon=False,
                            markerscale=3)
            for t in leg.get_texts():
                t.set_color(INK_2)
        ax.set_title(f"{v} removed — token classes", fontsize=10, color=INK_2)
        ax.set_xticklabels([]); ax.set_yticklabels([])

        ax = fig.add_subplot(2, 2, j + 3)
        style_ax(ax)
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=4, c=np.log10(d.freq),
                        cmap=seq_cmap(), linewidths=0, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("log10(count)", fontsize=8, color=INK_2)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_visible(False)
        ax.set_title(f"{v} removed — log frequency", fontsize=10, color=INK_2)
        ax.set_xticklabels([]); ax.set_yticklabels([])
    fig.suptitle("t-SNE (cos, perplexity 30) under the two frequency-removal "
                 "variants", fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "compare_tsne.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
