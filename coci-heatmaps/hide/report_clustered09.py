"""Second clustered-order variant (user rules 2026-09-02): same heatmaps as
report_alive.md / report_all.md, but clusters come from two threshold rules
instead of constrained linkage:

  1. Two components with co-CI r > 0.9 are in the same cluster (pairs
     processed in descending r, chains allowed).
  2. A component does not join a cluster if it has r <= 0.0 (or undefined r)
     with any existing member; two clusters only merge if every cross pair
     has r > 0. Such joins are skipped (counted per matrix).

No mean-CI-ratio constraint. Ordering as in report_clustered.py: clusters
placed by descending mean member CI (singletons land where the plain mean-CI
sort would put them), members within a cluster by descending mean CI.

Writes report_alive_clustered09.md / report_all_clustered09.md; figures in
hide/figures/<model>/<alive|all>_clustered09/.

Usage: python coci-heatmaps/hide/report_clustered09.py   (~2 min)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
OUT = HERE.parent / "old"  # reports moved into old/ 2026-09-02 (old vs new A/B decompositions)
CACHE = HERE / "cache"
sys.path.insert(0, str(HERE))
from report import MATS, MODELS, N_LAYERS  # noqa: E402

R_EDGE, R_BLOCK = 0.9, 0.0
CUT_COLOR = "#008300"


def heatmap(r, ids, sizes, title, path, cutoff=None,
            cmap="RdBu_r", vmin=-1, vmax=1):
    """report_clustered.heatmap + optional dashed alive/dead cutoff line."""
    n = len(ids)
    side = float(np.clip(2 + n / 220, 5, 18))
    fig, ax = plt.subplots(figsize=(side + 1.2, side), constrained_layout=True)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(r), cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    for b, s in zip(np.cumsum(sizes) - sizes, sizes):
        if s > 1:
            ax.add_patch(Rectangle((b - 0.5, b - 0.5), s, s, fill=False,
                                   edgecolor="black", lw=0.6))
    if cutoff is not None and 0 < cutoff < n:
        ax.axhline(cutoff - 0.5, color=CUT_COLOR, lw=1.2, ls="--")
        ax.axvline(cutoff - 0.5, color=CUT_COLOR, lw=1.2, ls="--")
        ax.text(n - 0.5, cutoff - 0.5, " alive | dead", color=CUT_COLOR,
                fontsize=8, va="bottom", ha="right")
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


def rule_clusters(r: np.ndarray, sel: np.ndarray) -> tuple[list, int]:
    """Greedy threshold clustering on the components in sel (local indices
    into sel are used internally). Returns (list of index-arrays, blocked)."""
    n = len(sel)
    Rf = np.nan_to_num(r[np.ix_(sel, sel)].astype(np.float64), nan=0.0)
    ii, jj = np.triu_indices(n, 1)
    e = Rf[ii, jj] > R_EDGE
    pairs = sorted(zip(Rf[ii[e], jj[e]], ii[e], jj[e]), reverse=True)
    of: dict[int, int] = {}
    members: dict[int, list] = {}
    nxt = blocked = 0
    for _, i, j in pairs:
        a, b = of.get(int(i)), of.get(int(j))
        if a is None and b is None:
            members[nxt] = [int(i), int(j)]
            of[int(i)] = of[int(j)] = nxt
            nxt += 1
        elif a == b:
            continue
        elif a is None or b is None:
            k, new = (b, int(i)) if a is None else (a, int(j))
            if Rf[new, members[k]].min() > R_BLOCK:
                members[k].append(new)
                of[new] = k
            else:
                blocked += 1
        else:
            if Rf[np.ix_(members[a], members[b])].min() > R_BLOCK:
                for q in members[b]:
                    of[q] = a
                members[a] += members.pop(b)
            else:
                blocked += 1
    return [sel[sorted(ms)] for ms in members.values()], blocked


def main() -> None:
    head = (
        "Same heatmaps as [report_{v}.md](report_{v}.md) (Pearson r of "
        "per-token CI; see there for data and definitions), but with "
        "threshold-rule clusters (alternative to "
        "[report_{v}_clustered.md](report_{v}_clustered.md)): **(1)** two "
        f"components with co-CI r > {R_EDGE} are in the same cluster (pairs "
        "processed in descending r, chains allowed); **(2)** a component "
        f"does not join a cluster if it has r ≤ {R_BLOCK} (or undefined r) "
        "with any existing member — two clusters only merge if every cross "
        "pair is > 0; such joins are skipped (count noted per matrix when "
        "> 0). No mean-CI-ratio constraint. Clusters are placed by "
        "descending mean member CI — singletons land where the plain "
        "mean-CI sort would put them — and members within a cluster are "
        "sorted by descending mean CI. **Black outlines** mark the "
        "multi-member clusters on the diagonal.{cut}")
    cut_note = (
        " The **dashed green line** marks the alive/dead cutoff: it sits at "
        "the alive count (harvest mean CI > 1e-6), i.e. exactly where the "
        "cutoff falls in plain mean-CI order; since clusters may mix alive "
        "and dead components (5 clusters do, across 4 pile matrices), a few "
        "components can sit on the wrong side of it in cluster order.")
    lines = {v: [
        f"# CI co-activation per matrix — {v} components, co-CI>0.9 clusters",
        "",
        head.replace("{v}", v).replace("{cut}", cut_note if v == "all" else ""),
        "",
    ] for v in ("alive", "all")}

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
                    multi, blocked = rule_clusters(r, sel)
                    in_multi = {int(c) for cl in multi for c in cl}
                    clusters = ([c[np.argsort(-m[c], kind="stable")]
                                 for c in multi]
                                + [np.array([c]) for c in sel
                                   if int(c) not in in_multi])
                    clusters.sort(key=lambda c: -m[c].mean())
                    order = np.concatenate(clusters)
                    sizes = [len(c) for c in clusters]
                    fig_dir = HERE / "figures" / model / f"{v}_clustered09"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"h{l}_{mat.replace('.', '_')}.png"
                    n, nmulti = len(order), sum(s > 1 for s in sizes)
                    title = (f"{model}  {mod}\nco-CI r(CI), {n} "
                             + ("alive " if v == "alive" else "")
                             + "components (co-CI>0.9 cluster order)")
                    heatmap(r[np.ix_(order, order)], order, sizes, title,
                            fig_dir / fname,
                            cutoff=len(alive) if v == "all" else None)
                    lines[v] += [
                        f"#### {mod} — {n} components, {nmulti} clusters "
                        f"with ≥ 2 members (largest {max(sizes)})"
                        + (f", {blocked} joins blocked" if blocked else ""),
                        "",
                        f"![{mod}](../hide/figures/{model}/{v}_clustered09/{fname})",
                        ""]
                    print(f"{model} {mod} [{v}]: {nmulti} multi-clusters, "
                          f"largest {max(sizes)}, blocked {blocked}")

    for v in ("alive", "all"):
        path = OUT / f"report_{v}_clustered09.md"
        path.write_text("\n".join(lines[v]), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
