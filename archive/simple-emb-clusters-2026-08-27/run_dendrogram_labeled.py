"""Ward dendrogram with per-group labels.

Redraws figures/dendrogram_ward.png: 80-leaf truncated dendrogram where each of
the 25 Ward clusters gets its own subtree color and, below the axis, a
token-type gloss + member count. Also writes clusters_ward.md (full listing,
in dendrogram left-to-right order) — the glosses in GLOSSES were written by
reading that listing.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from common import (GRID, HERE, INK_2, MUTED, embeddings_and_tokens,
                    frequencies, new_fig, save_fig, style_ax)
from run_clustering import CLUSTER_COLORS, K_FINE

# gloss per Ward cluster id (fcluster ids 0..24; deterministic across runs).
# Written by hand from clusters_ward.md.
GLOSSES: dict[int, str] = {
    0: "rare fragments & symbols (cone)",
    1: "people & names",
    2: "prepositions & directions",
    3: "auxiliaries & modals",
    4: "function words & punctuation",
    5: "rare suffixes & fragments",
    6: "##suffixes",
    7: "places",
    8: "light, sky & weather",
    9: "time & story nouns",
    10: "body & plant nouns",
    11: "emotions & virtues",
    12: "objects & treasures",
    13: "kings, parties & monsters",
    14: "animals & creatures",
    15: "adjectives: positive",
    16: "adjectives: states & moods",
    17: "short word-start fragments",
    18: "verbs: speech & cognition",
    19: "verbs: everyday actions",
    20: "sparkle & glow",
    21: "verbs: feeling & expression",
    22: "low-frequency words & fragments",
    23: "verbs: motion",
    24: "verbs: becoming & change",
}


def main():
    emb, tokens = embeddings_and_tokens()
    freq = frequencies(len(tokens))
    Xn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(Xn)

    Z = linkage(Xn, method="ward")
    clus = fcluster(Z, t=K_FINE, criterion="maxclust") - 1

    def members(node: int) -> np.ndarray:
        """Original point indices under a linkage node."""
        stack, out = [node], []
        while stack:
            k = stack.pop()
            if k < n:
                out.append(k)
            else:
                stack.extend(Z[k - n, :2].astype(int))
        return np.array(out)

    # Left-to-right leaf order of the truncated dendrogram; each truncated leaf
    # lies inside exactly one of the 25 clusters (80-leaf partition refines it).
    dd = dendrogram(Z, truncate_mode="lastp", p=80, no_plot=True)
    leaf_nodes = dd["leaves"]
    leaf_cluster = []
    for node in leaf_nodes:
        cids = np.unique(clus[members(node)])
        assert len(cids) == 1, (node, cids)
        leaf_cluster.append(int(cids[0]))
    cluster_order = list(dict.fromkeys(leaf_cluster))  # left-to-right, unique
    color_of = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                for i, c in enumerate(cluster_order)}

    # --- listing (read this to write GLOSSES) ---------------------------------
    lines = [f"# Ward k={K_FINE} — cluster listing (dendrogram left-to-right order)\n",
             "Per cluster: size, most frequent member tokens, tokens nearest "
             "(cosine) to the cluster centroid, and the gloss used in the "
             "dendrogram figure.\n"]
    for c in cluster_order:
        m = np.where(clus == c)[0]
        top = m[np.argsort(-freq[m])][:12]
        centroid = Xn[m].mean(0)
        near = m[np.argsort(-(Xn[m] @ centroid))][:8]
        lines += [f"**cluster {c}** (n={len(m)}) — gloss: *{GLOSSES.get(c, 'TODO')}*",
                  "- frequent: " + " ".join(f"`{tokens[i]}`" for i in top),
                  "- central: " + " ".join(f"`{tokens[i]}`" for i in near), ""]
    (HERE / "clusters_ward.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote clusters_ward.md ({len(cluster_order)} clusters, "
          f"{sum(c in GLOSSES for c in cluster_order)} glossed)")

    # --- figure ---------------------------------------------------------------
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

    # one label per cluster, centered under its span of leaves; text in a
    # darkened version of the subtree color so pale tab20 entries stay legible
    for c in cluster_order:
        xs = [5 + 10 * j for j, lc in enumerate(leaf_cluster) if lc == c]
        sz = int(np.sum(clus == c))
        # leading number = the cluster id shown in tsne_clusters_methods.png
        label = f"{c}: {GLOSSES.get(c, '?')}  ({sz})"
        r, g, b, _ = matplotlib.colors.to_rgba(color_of[c])
        ax.text(np.mean(xs), -0.15, label, rotation=90, rotation_mode="anchor",
                ha="right", va="center", fontsize=7.5, fontweight="bold",
                color=(0.55 * r, 0.55 * g, 0.55 * b))

    ax.set_title(f"Ward dendrogram (truncated to 80 leaves; colors + labels = the "
                 f"{K_FINE}-cluster cut, label = token type (group size))",
                 fontsize=10, color=INK_2)
    ax.set_ylabel("merge distance", fontsize=8, color=MUTED)
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, "dendrogram_ward.png")


if __name__ == "__main__":
    main()
