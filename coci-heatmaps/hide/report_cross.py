"""Cross-decomposition co-CI heatmaps: old vs newA, old vs newB, newA vs newB.

For every matrix, a rectangular heatmap of the Pearson r between the per-token
CI of a component of one decomposition (y axis) and a component of the other
(x axis), over the same 2.05M tokens as all other coci-heatmaps reports.
Alive components only. Ordering (diagonal-matched): y axis by descending mean
CI; each x component at its best-matching (max r >= MATCH_R) y component's
position, unmatched ones appended at the right by descending mean CI.
(clustered09_orders(), the previous both-axes ordering, is kept here for
report_cross_cosine.py.)

Inputs: cache/cross_partial_{newA,newB}.npz (from coci_cross_compute_modal.py),
cache/cross_alive.npz, and the per-decomposition caches for mean CI / the
orderings (coci_pile_4l.npz + mean_ci_pile_4l.npz for old, coci_{newA,newB}.npz).

Writes old-newA-newB/report_co_ci.md; figures in hide/figures/cross/<pair>/.

Usage: python coci-heatmaps/hide/report_cross.py   (~2 min)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE.parent / "old-newA-newB"
sys.path.insert(0, str(HERE))
from report import MATS  # noqa: E402
from report_clustered09 import rule_clusters  # noqa: E402

MODS = [f"h.{l}.{m}" for l in range(4) for m in MATS]
PAIRS = [("old", "newA"), ("old", "newB"), ("newA", "newB")]


def c09_order(r: np.ndarray, m: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Exactly the clustered09 ordering of report_clustered09.py/report_new09.py."""
    multi, _ = rule_clusters(r, sel)
    in_multi = {int(c) for cl in multi for c in cl}
    clusters = ([c[np.argsort(-m[c], kind="stable")] for c in multi]
                + [np.array([c]) for c in sel if int(c) not in in_multi])
    clusters.sort(key=lambda c: -m[c].mean())
    return np.concatenate(clusters)


def heatmap(r, ids_y, ids_x, name_y, name_x, title, path,
            cmap="RdBu_r", vmin=-1, vmax=1,
            note_y="clustered09 order", note_x="clustered09 order"):
    # square cells (aspect="equal"), figure sized proportionally to the
    # component counts; one shared inches-per-cell scale for both axes
    ny, nx = r.shape
    long = max(nx, ny)
    scale = float(np.clip(2 + long / 220, 4, 18)) / long
    if min(nx, ny) <= 120:  # room for tick labels, within the 18in cap
        scale = min(max(scale, 0.07), 18 / long)
    fig, ax = plt.subplots(figsize=(nx * scale + 2.4, ny * scale + 1.6),
                           constrained_layout=True)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(r), cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    fs = max(5, min(9, int(scale * 72 * 0.7)))
    if nx <= 120 and scale >= 0.045:
        ax.set_xticks(range(nx), ids_x, rotation=90, fontsize=fs)
        ax.tick_params(axis="x", length=0)
    if ny <= 120 and scale >= 0.045:
        ax.set_yticks(range(ny), ids_y, fontsize=fs)
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"{name_x} components ({note_x})", fontsize=9)
    ax.set_ylabel(f"{name_y} components ({note_y})", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def clustered09_orders(alive):
    """Each decomposition's alive components in its report_alive_clustered09.md
    order, per matrix. Returns (ids, order_pos): ids[name][mod] = global
    component ids in display order; order_pos[name][mod] = the same order as
    positions into the ascending-alive-id axis (the dump/cache column order)."""
    print("computing clustered09 orders ...", flush=True)
    src = {"old": (np.load(CACHE / "coci_pile_4l.npz"),
                   np.load(CACHE / "mean_ci_pile_4l.npz")),
           "newA": (np.load(CACHE / "coci_newA.npz"),) * 2,
           "newB": (np.load(CACHE / "coci_newB.npz"),) * 2}
    ids = {n: {} for n in src}
    order_pos = {n: {} for n in src}
    for name, (rz, mz) in src.items():
        for mod in MODS:
            r = rz[f"{mod}|r"].astype(np.float32)
            m = mz[mod] if name == "old" else mz[f"{mod}|mean"]
            sel = np.argsort(-m, kind="stable")
            sel = sel[m[sel] > 1e-6]
            order = c09_order(r, m, sel)                    # global comp ids
            a = alive[f"{name}|{mod}"]                      # ascending ids
            assert len(order) == len(a) and set(order) == set(a)
            ids[name][mod] = order
            order_pos[name][mod] = np.searchsorted(a, order)  # dump columns
    return ids, order_pos


MATCH_R = 0.3  # min r for an x component to count as matched to a y component


def ci_orders(alive):
    """Plain descending-mean-CI order per decomposition. Returns (ids,
    order_pos, ci_dump): ids[name][mod] = global component ids in CI order;
    order_pos[name][mod] = that order as positions into the ascending-alive-id
    (dump/cache) axis; ci_dump[name][mod] = mean CI in dump order."""
    src = {"old": np.load(CACHE / "mean_ci_pile_4l.npz"),
           "newA": np.load(CACHE / "coci_newA.npz"),
           "newB": np.load(CACHE / "coci_newB.npz")}
    ids = {n: {} for n in src}
    order_pos = {n: {} for n in src}
    ci_dump = {n: {} for n in src}
    for name, mz in src.items():
        for mod in MODS:
            m = mz[mod] if name == "old" else mz[f"{mod}|mean"]
            a = alive[f"{name}|{mod}"]
            ma = m[a]
            pos = np.argsort(-ma, kind="stable")
            ids[name][mod] = a[pos]
            order_pos[name][mod] = pos
            ci_dump[name][mod] = ma
    return ids, order_pos, ci_dump


def cross_stats():
    """Load the cross partials -> (alive, stats, G, T); stats/G in the
    ascending-alive-index (dump) axis order."""
    pA = np.load(CACHE / "cross_partial_newA.npz")
    pB = np.load(CACHE / "cross_partial_newB.npz")
    alive = np.load(CACHE / "cross_alive.npz")
    T = float(pA["T"])
    assert float(pB["T"]) == T

    # per-decomposition mean/std over the f16-consistent series, per matrix
    stats = {}
    for name, part in (("old", pA), ("newA", pA), ("newB", pB)):
        stats[name] = {}
        for mod in MODS:
            s1, s2 = part[f"{mod}|S1_{name}"], part[f"{mod}|S2_{name}"]
            mu = s1 / T
            sd = np.sqrt(np.clip(s2 / T - mu**2, 0, None))
            stats[name][mod] = (mu, sd)
    G = {("old", "newA"): pA, ("old", "newB"): pB, ("newA", "newB"): pB}
    return alive, stats, G, T


def pair_r(stats, G, T, ny_, nx_, mod):
    """Cross co-CI Pearson r in dump order (NaN where undefined)."""
    muy, sdy = stats[ny_][mod]
    mux, sdx = stats[nx_][mod]
    g = G[(ny_, nx_)][f"{mod}|G_{ny_}_{nx_}"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (g / T - np.outer(muy, mux)) / np.outer(sdy, sdx)
    r[~np.isfinite(r)] = np.nan
    return r


def diagonal_orders(alive, stats, G, T, match_r=MATCH_R):
    """The report_co_ci.md diagonal-matched ordering for every pair x matrix,
    derived from co-CI r: rows by descending mean CI, columns at their
    best-matching (max r >= match_r) row's position (ties by own mean CI),
    unmatched appended by mean CI. Pass match_r=-np.inf to place EVERY column
    at its argmax row (no unmatched tail). Returns orders[(ny, nx, mod)] =
    (row_pos, col_pos, n_matched), positions into the dump axes."""
    _, order_pos, ci_dump = ci_orders(alive)
    orders = {}
    for ny_, nx_ in PAIRS:
        for mod in MODS:
            r = pair_r(stats, G, T, ny_, nx_, mod)[order_pos[ny_][mod]]
            s = np.nan_to_num(r, nan=-1.0)
            best, smax = s.argmax(0), s.max(0)
            cx = ci_dump[nx_][mod]
            mi = np.flatnonzero(smax >= match_r)
            ui = np.flatnonzero(smax < match_r)
            mi = mi[np.lexsort((-cx[mi], best[mi]))]
            ui = ui[np.argsort(-cx[ui], kind="stable")]
            col = np.concatenate([mi, ui]).astype(np.intp)
            orders[(ny_, nx_, mod)] = (order_pos[ny_][mod], col, len(mi))
    return orders


def main() -> None:
    alive, stats, G, T = cross_stats()
    orders = diagonal_orders(alive, stats, G, T)

    lines = [
        "# Cross-decomposition CI co-activation: old vs newA vs newB",
        "",
        "For each matrix of the pile_4l target, heatmaps of the **Pearson r "
        "between the per-token causal importance of a component of one "
        "decomposition and a component of another** — pairs (old, newA), "
        "(old, newB), (newA, newB); the first-named decomposition is the "
        "y axis. Same CI measure (lower_leaky / clip(preact, 0, 1), "
        "continuous sampling) and the same 4,000 cached Pile rows (2.05M "
        "tokens) as the per-decomposition reports. **old** = `s-55ea3f9b` "
        "(the paper's decomposition), **newA** = `p-8383f5e5`, **newB** = "
        "`p-4d9a6a12` (see CLAUDE.md \"New 800k-step decompositions\").",
        "",
        "Alive components only (old: harvest mean CI > 1e-6; newA/newB: "
        "sample mean CI > 1e-6). **Ordering, chosen to make matches "
        "diagonal:** the y axis is that decomposition's alive components by "
        "descending mean CI; each x component is placed at the position of "
        f"its best-matching y component (max r, when that r ≥ {MATCH_R}), "
        "ties broken by its own mean CI — so shared mechanisms line up "
        "along a diagonal band (with vertical stripes where several x "
        "components match one y component). x components with no match "
        f"≥ {MATCH_R} follow at the right, by descending mean CI. "
        "**Gray** = zero CI variance in the sample (r undefined).",
        "",
    ]

    prev_blk = None
    for l in range(4):
        for mat in MATS:
            mod = f"h.{l}.{mat}"
            # rule between matrices; double rule at attn <-> MLP transitions
            hr = '<hr style="height:10px;background:#555;border:none;">'
            blk = mat.split(".")[0]
            if prev_blk is not None:
                lines += [hr, ""] * (1 if blk == prev_blk else 2)
            prev_blk = blk
            if mat == MATS[0]:
                lines += [f"### Layer {l}", ""]
            lines += [f"#### {mod}", ""]
            for ny_, nx_ in PAIRS:
                row_pos, col, n_matched = orders[(ny_, nx_, mod)]
                r = pair_r(stats, G, T, ny_, nx_, mod)[row_pos][:, col]
                ids_y = alive[f"{ny_}|{mod}"][row_pos]
                ids_x = alive[f"{nx_}|{mod}"][col]
                pair = f"{ny_}_{nx_}"
                fig_dir = HERE / "figures" / "cross" / pair
                fig_dir.mkdir(parents=True, exist_ok=True)
                fname = f"h{l}_{mat.replace('.', '_')}.png"
                title = (f"{ny_} × {nx_}   {mod}\ncross co-CI r(CI), "
                         f"{r.shape[0]} × {r.shape[1]} alive components "
                         f"({n_matched} matched at r ≥ {MATCH_R})")
                heatmap(r, ids_y, ids_x, ny_, nx_, title,
                        fig_dir / fname,
                        note_y="mean CI order",
                        note_x="best-match position; unmatched right, by mean CI")
                lines += [f"![{mod} {pair}](../hide/figures/cross/{pair}/{fname})",
                          ""]
                print(f"{mod} {pair}: {r.shape[0]}x{r.shape[1]}, "
                      f"max r {np.nanmax(r):.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    path = OUT / "report_co_ci.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
