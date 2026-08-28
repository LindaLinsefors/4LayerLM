"""Print all tokens of one Ward cluster, grouped by its subtrees as displayed
in the truncated (80-leaf) dendrogram figure.

Usage: python inspect_cluster.py [variant] [cluster_id]   (default pc1 1)
Subgroups are printed in dendrogram left-to-right order; tokens within a
subgroup in descending corpus frequency.
"""

import sys

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from common import processed
from run_clustering import K_FINE

variant = sys.argv[1] if len(sys.argv) > 1 else "pc1"
cluster_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

d = processed(variant)
n = len(d.tokens)

Z = linkage(d.Xn, method="ward")
clus = fcluster(Z, t=K_FINE, criterion="maxclust") - 1


def members(node: int) -> np.ndarray:
    stack, out = [node], []
    while stack:
        k = stack.pop()
        out.append(k) if k < n else stack.extend(Z[k - n, :2].astype(int))
    return np.array(out)


dd = dendrogram(Z, truncate_mode="lastp", p=80, no_plot=True)
sub_leaves = [nd for nd in dd["leaves"] if clus[members(nd)[0]] == cluster_id]

print(f"[{variant}] Ward cluster {cluster_id} (n={np.sum(clus == cluster_id)}), "
      f"{len(sub_leaves)} subgroups in the 80-leaf dendrogram:\n")
for i, node in enumerate(sub_leaves):
    m = members(node)
    m = m[np.argsort(-d.freq[m])]
    print(f"--- subgroup {i + 1} (n={len(m)}) ---")
    print(" ".join(d.tokens[j] for j in m))
    print()
