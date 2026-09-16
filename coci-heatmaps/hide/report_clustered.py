"""Clustered-order variant of report.py: same heatmaps, but the component
order trades off mean-CI sorting against CI-similarity clustering.

Ordering: constrained agglomerative clustering (scipy, complete linkage) on
d = 1 - r(CI); pairs whose harvest mean CIs differ by more than a factor 2
(and pairs involving a zero-variance component) get d = BIG, so with complete
linkage no flat cluster can ever contain such a pair. Flat clusters at
complete-linkage distance <= 0.7, i.e. every within-cluster pair has
r >= 0.3 AND mean CIs within 2x. Clusters are placed by descending mean of
member mean CI (singletons land where plain CI sorting would put them);
members within a cluster by descending mean CI. Multi-member clusters are
outlined in black on the diagonal.

Writes report_alive_clustered.md / report_all_clustered.md; figures in
hide/figures/<model>/<alive|all>_clustered/.

Usage: python coci-heatmaps/hide/report_clustered.py   (~2 min)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).parent
OUT = HERE.parent / "old"  # reports moved into old/ 2026-09-02 (old vs new A/B decompositions)
CACHE = HERE / "cache"
sys.path.insert(0, str(HERE))
from report import MATS, MODELS, N_LAYERS, TOKENS_DESC

BIG = 1e6          # forbidden-pair distance (CI ratio > 2 or undefined r)
CLUSTER_T = 0.7    # complete-linkage cut: all within-cluster pairs r >= 0.3


def clustered_order(r: np.ndarray, m: np.ndarray, sel: np.ndarray):
    """Return (order, block sizes) for the components in sel."""
    n = len(sel)
    if n < 2:
        return sel, [1] * n
    d = 1.0 - r[np.ix_(sel, sel)].astype(np.float64)
    d = np.clip(d, 0, 2)
    ms = m[sel]
    ratio_bad = np.maximum.outer(ms, ms) > 2 * np.minimum.outer(ms, ms)
    d[ratio_bad | ~np.isfinite(d)] = BIG
    np.fill_diagonal(d, 0)
    labels = fcluster(linkage(squareform(d, checks=False), method="complete"),
                      t=CLUSTER_T, criterion="distance")
    clusters = [sel[labels == k] for k in np.unique(labels)]
    clusters = [c[np.argsort(-m[c], kind="stable")] for c in clusters]
    clusters.sort(key=lambda c: -m[c].mean())
    return np.concatenate(clusters), [len(c) for c in clusters]


def heatmap(r, ids, sizes, title, path):
    n = len(ids)
    side = float(np.clip(2 + n / 220, 5, 18))
    fig, ax = plt.subplots(figsize=(side + 1.2, side), constrained_layout=True)
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(r), cmap=cm, vmin=-1, vmax=1,
                   interpolation="nearest")
    for b, s in zip(np.cumsum(sizes) - sizes, sizes):
        if s > 1:
            ax.add_patch(Rectangle((b - 0.5, b - 0.5), s, s, fill=False,
                                   edgecolor="black", lw=0.6))
    if n <= 120:
        fs = max(5, min(9, int((side - 1) * 72 / n * 0.6)))
        ax.set_xticks(range(n), ids, rotation=90, fontsize=fs)
        ax.set_yticks(range(n), ids, fontsize=fs)
        ax.tick_params(length=0)
    else:
        ax.set_xlabel("rank (clusters by descending mean CI)", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    lines = {v: [
        f"# CI co-activation per matrix — {t} components, CI-clustered order",
        "",
        "Same heatmaps as " + f"[report_{v}.md](report_{v}.md)" + " (Pearson r "
        "of per-token CI; see there for data and definitions), but the "
        "component order trades off mean-CI sorting against CI-similarity "
        "clustering: constrained complete-linkage clustering on d = 1 − r, "
        "with pairs whose harvest mean CIs differ by **more than a factor 2** "
        "(or with undefined r) forbidden (d = ∞) — so no cluster ever mixes "
        "components of >2× different average CI. Flat clusters at complete-"
        "linkage d ≤ 0.7 (⇒ every within-cluster pair has r ≥ 0.3). Clusters "
        "are placed by descending mean member CI — singletons land where the "
        "plain mean-CI sort would put them — and members within a cluster are "
        "sorted by descending mean CI. **Black outlines** mark the multi-"
        "member clusters on the diagonal.",
        "",
    ] for v, t in (("alive", "alive"), ("all", "all"))}

    for model in MODELS:
        stats = np.load(CACHE / f"coci_{model}.npz")
        mci = np.load(CACHE / f"mean_ci_{model}.npz")
        for v in lines:
            lines[v] += [f"## {model}", ""]
        for l in range(N_LAYERS[model]):
            for v in lines:
                lines[v] += [f"### Layer {l}", ""]
            for mat in MATS:
                mod = f"h.{l}.{mat}"
                r = stats[f"{mod}|r"].astype(np.float32)
                m = mci[mod]
                order_all = np.argsort(-m, kind="stable")
                alive = order_all[m[order_all] > 1e-6]
                for v, sel in (("alive", alive), ("all", order_all)):
                    order, sizes = clustered_order(r, m, sel)
                    fig_dir = HERE / "figures" / model / f"{v}_clustered"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"h{l}_{mat.replace('.', '_')}.png"
                    n, nmulti = len(order), sum(s > 1 for s in sizes)
                    title = (f"{model}  {mod}\nco-CI r(CI), {n} "
                             + ("alive " if v == "alive" else "")
                             + "components (CI-clustered order)")
                    heatmap(r[np.ix_(order, order)], order, sizes, title,
                            fig_dir / fname)
                    lines[v] += [
                        f"#### {mod} — {n} components, {nmulti} clusters "
                        f"with ≥ 2 members (largest {max(sizes)})", "",
                        f"![{mod}](../hide/figures/{model}/{v}_clustered/{fname})",
                        ""]
                    print(f"{model} {mod} [{v}]: {nmulti} multi-clusters, "
                          f"largest {max(sizes)}")

    for v in ("alive", "all"):
        path = OUT / f"report_{v}_clustered.md"
        path.write_text("\n".join(lines[v]), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
