"""Clustering of the de-frequencied alive-token embeddings, both variants.

All methods run on the row-normalized processed embeddings Xn (euclidean
distance monotone in cosine distance: |x-y|^2 = 2(1 - cos)), except where
noted.

Methods:
  KMeans           -- k sweep 2..40, silhouette + Davies-Bouldin model selection
  Ward             -- agglomerative, labeled dendrogram, cut at chosen k
  GMM (PCA-50)     -- BIC model selection over k
  Spectral         -- k-NN graph affinity
  HDBSCAN          -- density-based, on Xn (192-d) and on the 2-d t-SNE map

Per variant: figures/<variant>/{model_selection,dendrogram_ward,
tsne_clusters_kmeans,tsne_clusters_methods,method_agreement}.png; listings
clusters_{kmeans,ward,hdbscan}_<variant>.md; cache/<variant>/{labels.npz,
metrics.json}. Dendrogram cluster labels are automatic (top frequent tokens).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from common import (GRID, HERE, INK, INK_2, MUTED, SEQ_RAMP, VARIANTS,
                    cache_dir, new_fig, processed, save_fig, seq_cmap, style_ax)

K_RANGE = range(2, 41)
K_FINE = 25  # finer clustering used for interpretation & method comparison
TSNE_MAIN = "cos30"  # must exist in cache/<variant>/ (run run_tsne.py first)

# 25 distinct cluster colors: tab20 + every 4th of tab20b.
CLUSTER_COLORS = [matplotlib.colormaps["tab20"](i) for i in range(20)] + \
                 [matplotlib.colormaps["tab20b"](i) for i in (0, 4, 8, 12, 16)]


def cluster_scatter(ax, xy, labels, title):
    """t-SNE scatter colored by cluster, with the cluster id at each centroid
    (direct labels, so identity does not ride on color alone). Noise = -1, gray."""
    style_ax(ax)
    for c in sorted(set(labels)):
        m = labels == c
        color = "#c3c2b7" if c == -1 else CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(xy[m, 0], xy[m, 1], s=3, c=[color], linewidths=0, alpha=0.85)
        if c != -1:
            cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
            ax.text(cx, cy, str(c), fontsize=8, color=INK, ha="center", va="center",
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_title(title, fontsize=10, color=INK_2)
    ax.set_xticklabels([]); ax.set_yticklabels([])


def listing(path, method_name, labels, tokens, freq, Xn):
    """Markdown listing: per cluster, most frequent tokens + nearest to centroid."""
    lines = [f"# {method_name} — cluster listing\n",
             "Per cluster: size, the most frequent member tokens, and the tokens "
             "nearest (cosine) to the cluster centroid.\n"]
    for c in sorted(set(labels), key=lambda c: (c == -1, -np.sum(labels == c))):
        m = np.where(labels == c)[0]
        top = m[np.argsort(-freq[m])][:12]
        name = "noise" if c == -1 else f"cluster {c}"
        lines.append(f"**{name}** (n={len(m)})")
        lines.append("- frequent: " + " ".join(f"`{tokens[i]}`" for i in top))
        if c != -1:
            centroid = Xn[m].mean(0)
            near = m[np.argsort(-(Xn[m] @ centroid))][:8]
            lines.append("- central: " + " ".join(f"`{tokens[i]}`" for i in near))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.relative_to(HERE.parent)}")


def labeled_dendrogram(Z, clus, tokens, freq, variant):
    """80-leaf truncated dendrogram; each K_FINE-cluster subtree colored and
    labeled below the axis with its id, top-3 frequent member tokens, size."""
    n = len(tokens)

    def members(node: int) -> np.ndarray:
        stack, out = [node], []
        while stack:
            k = stack.pop()
            out.append(k) if k < n else stack.extend(Z[k - n, :2].astype(int))
        return np.array(out)

    dd = dendrogram(Z, truncate_mode="lastp", p=80, no_plot=True)
    leaf_cluster = [int(clus[members(nd)[0]]) for nd in dd["leaves"]]
    cluster_order = list(dict.fromkeys(leaf_cluster))  # left-to-right, unique
    color_of = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                for i, c in enumerate(cluster_order)}

    def link_color(node: int) -> str:
        cids = np.unique(clus[members(node)])
        return matplotlib.colors.to_hex(color_of[int(cids[0])]) if len(cids) == 1 else GRID

    fig = new_fig(figsize=(13.5, 7))
    ax = fig.add_subplot(111)
    style_ax(ax, square=False)
    ax.grid(False)
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=80, no_labels=True,
               link_color_func=link_color)
    ax.set_xticks([])

    for c in cluster_order:
        xs = [5 + 10 * j for j, lc in enumerate(leaf_cluster) if lc == c]
        m = np.where(clus == c)[0]
        top = " ".join(tokens[i] for i in m[np.argsort(-freq[m])][:3])
        label = f"{c}: {top}  ({len(m)})"
        r, g, b, _ = matplotlib.colors.to_rgba(color_of[c])
        ax.text(np.mean(xs), -0.15, label, rotation=90, rotation_mode="anchor",
                ha="right", va="center", fontsize=7.5, fontweight="bold",
                color=(0.55 * r, 0.55 * g, 0.55 * b))

    ax.set_title(f"Ward dendrogram, {variant} removed (80 leaves; colors = "
                 f"{K_FINE}-cluster cut, label = id: top tokens (size))",
                 fontsize=10, color=INK_2)
    ax.set_ylabel("merge distance", fontsize=8, color=MUTED)
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, "dendrogram_ward.png", variant)
    plt.close(fig)


def run_variant(variant: str):
    print(f"=== variant {variant} ===", flush=True)
    d = processed(variant)
    tokens, freq, classes, Xn = d.tokens, d.freq, d.classes, d.Xn
    xy = np.load(cache_dir(variant) / f"tsne_{TSNE_MAIN}.npy")

    # --- KMeans sweep --------------------------------------------------------
    print("KMeans sweep ...", flush=True)
    sil, db, km_labels = [], [], {}
    for k in K_RANGE:
        lab = KMeans(k, n_init=5, random_state=0).fit_predict(Xn)
        km_labels[k] = lab
        sil.append(silhouette_score(Xn, lab))
        db.append(davies_bouldin_score(Xn, lab))
    k_opt = list(K_RANGE)[int(np.argmax(sil))]
    print(f"  silhouette-optimal k = {k_opt}")

    # --- GMM / BIC on PCA-50 -------------------------------------------------
    print("GMM BIC sweep ...", flush=True)
    Xp = PCA(50, random_state=0).fit_transform(Xn)
    bic = []
    gmm_labels = {}
    for k in K_RANGE:
        g = GaussianMixture(k, covariance_type="full", random_state=0,
                            n_init=1).fit(Xp)
        bic.append(g.bic(Xp))
        gmm_labels[k] = g.predict(Xp)
    k_bic = list(K_RANGE)[int(np.argmin(bic))]
    print(f"  BIC-optimal k = {k_bic}")

    # --- Ward ----------------------------------------------------------------
    print("Ward linkage ...", flush=True)
    Z = linkage(Xn, method="ward")
    ward_fine = fcluster(Z, t=K_FINE, criterion="maxclust") - 1

    # --- Spectral ------------------------------------------------------------
    print("Spectral ...", flush=True)
    spec_fine = SpectralClustering(K_FINE, affinity="nearest_neighbors",
                                   n_neighbors=20, assign_labels="cluster_qr",
                                   random_state=0).fit_predict(Xn)

    # --- HDBSCAN -------------------------------------------------------------
    print("HDBSCAN ...", flush=True)
    hdb_high = HDBSCAN(min_cluster_size=15).fit_predict(Xn)
    hdb_tsne = HDBSCAN(min_cluster_size=25).fit_predict(xy)
    for name, lab in [("HDBSCAN on 192-d", hdb_high), ("HDBSCAN on t-SNE", hdb_tsne)]:
        nc = len(set(lab)) - (-1 in lab)
        print(f"  {name}: {nc} clusters, {np.mean(lab == -1):.1%} noise")

    # --- figures -------------------------------------------------------------
    # model selection curves
    fig = new_fig(figsize=(12, 3.6))
    curves = [(list(K_RANGE), sil, "KMeans silhouette (higher better)", k_opt),
              (list(K_RANGE), db, "KMeans Davies-Bouldin (lower better)", None),
              (list(K_RANGE), bic, "GMM BIC on PCA-50 (lower better)", k_bic)]
    for j, (ks, ys, title, kmark) in enumerate(curves):
        ax = fig.add_subplot(1, 3, j + 1)
        style_ax(ax, square=False)
        ax.plot(ks, ys, color=SEQ_RAMP[7], linewidth=2)
        if kmark is not None:
            ax.axvline(kmark, color="#e34948", linewidth=1, linestyle="--")
            ax.text(kmark + 0.5, np.mean(ax.get_ylim()), f"k={kmark}",
                    color="#e34948", fontsize=8)
        ax.set_title(title, fontsize=9, color=INK_2)
        ax.set_xlabel("k", fontsize=8, color=MUTED)
    fig.tight_layout()
    save_fig(fig, "model_selection.png", variant)
    plt.close(fig)

    # labeled dendrogram
    labeled_dendrogram(Z, ward_fine, tokens, freq, variant)

    # kmeans at k_opt and K_FINE on the t-SNE map
    fig = new_fig(figsize=(13, 6.5))
    cluster_scatter(fig.add_subplot(1, 2, 1), xy, km_labels[k_opt],
                    f"KMeans k={k_opt} (silhouette-optimal)")
    cluster_scatter(fig.add_subplot(1, 2, 2), xy, km_labels[K_FINE],
                    f"KMeans k={K_FINE} (fine)")
    fig.suptitle(f"KMeans clusters on the t-SNE map ({variant} removed)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "tsne_clusters_kmeans.png", variant)
    plt.close(fig)

    # other methods
    fig = new_fig(figsize=(13, 13))
    cluster_scatter(fig.add_subplot(2, 2, 1), xy, ward_fine, f"Ward, cut at k={K_FINE}")
    cluster_scatter(fig.add_subplot(2, 2, 2), xy, spec_fine, f"Spectral, k={K_FINE}")
    cluster_scatter(fig.add_subplot(2, 2, 3), xy, gmm_labels[K_FINE],
                    f"GMM (PCA-50), k={K_FINE}")
    cluster_scatter(fig.add_subplot(2, 2, 4), xy, hdb_tsne,
                    "HDBSCAN on the t-SNE map (gray = noise)")
    fig.suptitle(f"Other clustering methods on the t-SNE map ({variant} removed)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    save_fig(fig, "tsne_clusters_methods.png", variant)
    plt.close(fig)

    # --- method agreement + class alignment ----------------------------------
    methods = {f"KMeans {K_FINE}": km_labels[K_FINE], f"Ward {K_FINE}": ward_fine,
               f"Spectral {K_FINE}": spec_fine, f"GMM {K_FINE}": gmm_labels[K_FINE],
               "HDBSCAN(t-SNE)": hdb_tsne}
    names = list(methods)
    ari = np.array([[adjusted_rand_score(methods[a], methods[b]) for b in names]
                    for a in names])
    fig = new_fig(figsize=(5.6, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(ari, cmap=seq_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right",
                  fontsize=8, color=INK_2)
    ax.set_yticks(range(len(names)), names, fontsize=8, color=INK_2)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{ari[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if ari[i, j] > 0.55 else INK)
    ax.set_title("Adjusted Rand index between methods", fontsize=10, color=INK_2)
    fig.colorbar(im, shrink=0.8).ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    save_fig(fig, "method_agreement.png", variant)
    plt.close(fig)

    nmi_class = {name: normalized_mutual_info_score(classes, lab)
                 for name, lab in methods.items()}
    print("NMI with token classes:", {k: round(v, 3) for k, v in nmi_class.items()})

    # --- listings + caches ---------------------------------------------------
    listing(HERE.parent / f"clusters_kmeans_{variant}.md", f"KMeans k={K_FINE} ({variant})",
            km_labels[K_FINE], tokens, freq, Xn)
    listing(HERE.parent / f"clusters_ward_{variant}.md", f"Ward k={K_FINE} ({variant})",
            ward_fine, tokens, freq, Xn)
    listing(HERE.parent / f"clusters_hdbscan_{variant}.md",
            f"HDBSCAN on t-SNE map ({variant})", hdb_tsne, tokens, freq, Xn)

    np.savez(cache_dir(variant) / "labels.npz", kmeans_opt=km_labels[k_opt],
             kmeans_fine=km_labels[K_FINE], ward_fine=ward_fine,
             spectral_fine=spec_fine, gmm_fine=gmm_labels[K_FINE],
             gmm_bic=gmm_labels[k_bic], hdbscan_192d=hdb_high, hdbscan_tsne=hdb_tsne)
    metrics = dict(
        k_range=list(K_RANGE), silhouette=[float(s) for s in sil],
        davies_bouldin=[float(x) for x in db], bic=[float(b) for b in bic],
        k_opt=int(k_opt), k_bic=int(k_bic), k_fine=K_FINE,
        hdbscan_192d=dict(n_clusters=int(len(set(hdb_high)) - (-1 in hdb_high)),
                          noise_frac=float(np.mean(hdb_high == -1))),
        hdbscan_tsne=dict(n_clusters=int(len(set(hdb_tsne)) - (-1 in hdb_tsne)),
                          noise_frac=float(np.mean(hdb_tsne == -1))),
        ari={f"{a} vs {b}": float(ari[i, j]) for i, a in enumerate(names)
             for j, b in enumerate(names) if i < j},
        nmi_with_token_classes={k: float(v) for k, v in nmi_class.items()},
    )
    (cache_dir(variant) / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"wrote cache/{variant}/metrics.json")


if __name__ == "__main__":
    for variant in VARIANTS:
        run_variant(variant)
