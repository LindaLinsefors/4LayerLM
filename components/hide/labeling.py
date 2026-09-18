"""Greedy non-overlapping token labels for the v_dot histogram panels.

Labels are rotated 90 deg, one line height wide, anchored above the histogram
bars at the token's x. Placement is greedy in priority order (extreme-tail
tokens - top 0.5% by |x - median| - first in extremity order, everything else
in corpus-count order): each label takes the lowest free spot at its x - just
above the tallest bar its text spans, or stacked on top of already-placed
labels there (skyline packing with bounding-box collision in log-y space) -
and is dropped only when even the stack would exceed the reserved headroom
(the y-limit is pre-extended so bars keep ~BAR_FRAC of the panel height).
Hence an isolated token is always labeled, and label count is bounded by the
available panel area, not by one label per x-slot.
"""
from bisect import bisect_left

import numpy as np

INK = "#39485E"
BAR_FRAC = 0.70  # fraction of the final panel height kept for the bars


def _short(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars - 1] + "\N{HORIZONTAL ELLIPSIS}"


def label_order(x: np.ndarray, counts: np.ndarray, tail_q: float = 0.995) -> np.ndarray:
    """Priority order for labeling: tail tokens by extremity, then bulk by count."""
    ext = np.abs(x - np.median(x))
    tail = ext > np.quantile(ext, tail_q)
    by_ext = np.argsort(-ext)
    rest = np.lexsort((-ext, -counts))
    return np.concatenate([by_ext[tail[by_ext]], rest[~tail[rest]]])


def add_hist_labels(ax, fig, x, names, order, weights=None, bins=200,
                    fontsize=6, max_chars=12) -> int:
    """Annotate as many tokens as fit without overlapping text; returns count.

    Call after fig.tight_layout() + fig.canvas.draw() so transforms are final.
    x/names/order index the same token set; weights as passed to ax.hist.
    """
    hist, edges = np.histogram(x, bins=bins, weights=weights)
    # label width = one rotated line height, converted to data units
    a, b = ax.transData.transform([(0.0, 1.0), (1.0, 1.0)])[:, 0]
    wid = 1.15 * fontsize / 72 * fig.dpi / abs(b - a)
    # pre-extend the y-limit: bars keep BAR_FRAC of the height, rest = headroom
    ybot, ytop = ax.get_ylim()
    lb = np.log10(ybot)
    decades = (np.log10(ytop) - lb) / BAR_FRAC
    top_log = lb + decades
    ax.set_ylim(top=10.0 ** top_log)
    ax_px = ax.get_position().height * fig.get_size_inches()[1] * fig.dpi
    dec_per_px = decades / ax_px
    pad = 2.0 * fontsize / 72 * fig.dpi * dec_per_px  # 2 pt gap between stacked labels
    xs: list[float] = []                      # placed label x's, kept sorted
    ivs: list[tuple[float, float]] = []       # (lo, hi) log10-y, aligned to xs
    n = 0
    for i in order:
        xi = float(x[i])
        s = names[i]
        if len(s) > max_chars:
            s = s[:max_chars - 1] + "\N{HORIZONTAL ELLIPSIS}"
        h = (2 + 0.65 * len(s)) * fontsize / 72 * fig.dpi * dec_per_px
        # base: above the tallest bar the rotated text horizontally spans
        j0 = np.searchsorted(edges, xi - wid / 2, side="right") - 1
        j1 = np.searchsorted(edges, xi + wid / 2, side="left")
        base = max(float(hist[max(j0, 0):min(j1, len(hist))].max(initial=0.0)),
                   ybot) * 1.3
        y = np.log10(base)
        # lowest free spot at this x: skip over the labels the text would hit
        k0, k1 = bisect_left(xs, xi - wid), bisect_left(xs, xi + wid)
        for lo, hi in sorted(ivs[k0:k1]):
            if y + h <= lo - pad:
                break
            y = max(y, hi + pad)
        if y + h > top_log:
            continue  # even the stack is full here — token stays unlabeled
        k = bisect_left(xs, xi)
        xs.insert(k, xi)
        ivs.insert(k, (y, y + h))
        n += 1
        ax.annotate(s, (xi, 10.0 ** y), rotation=90, fontsize=fontsize,
                    color=INK, ha="center", va="bottom", clip_on=False)
    return n


def add_scatter_labels(ax, fig, x, y, names, counts, fontsize=6,
                       max_chars=12, tail_q=0.995) -> int:
    """Dense non-overlapping labels for a scatter plot; returns label count.

    Same rules as the histogram version, in 2D: horizontal labels tried right /
    left / above / below of their point (2 px gap), greedy in priority order
    (top 0.5% by axes-normalized distance from the cloud median first, in
    distance order; rest by corpus count), rejected only if every position
    would overlap an already-placed label or leave the axes. Collision via a
    uniform spatial grid in display px. Call after tight_layout + canvas.draw.
    """
    P = ax.transData.transform(np.column_stack([x, y]))
    (ax0, ay0), (ax1, ay1) = ax.get_window_extent().get_points()
    lh = 1.15 * fontsize / 72 * fig.dpi
    gap = 2.0

    # priority: 2D extremity tail first, then corpus count
    z = (P - np.median(P, axis=0)) / [ax1 - ax0, ay1 - ay0]
    dist = np.hypot(z[:, 0], z[:, 1])
    tail = dist > np.quantile(dist, tail_q)
    by_d = np.argsort(-dist)
    rest = np.lexsort((-dist, -counts))
    order = np.concatenate([by_d[tail[by_d]], rest[~tail[rest]]])

    cell = 48.0
    grid: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}

    def cells(b):
        return [(i, j)
                for i in range(int(b[0] // cell), int(b[2] // cell) + 1)
                for j in range(int(b[1] // cell), int(b[3] // cell) + 1)]

    def collides(b):
        return any(b[0] < a[2] and a[0] < b[2] and b[1] < a[3] and a[1] < b[3]
                   for c in cells(b) for a in grid.get(c, ()))

    n = 0
    for i in order:
        px, py = P[i]
        if not (ax0 <= px <= ax1 and ay0 <= py <= ay1):
            continue
        s = _short(names[i], max_chars)
        w = (2 + 0.65 * len(s)) * fontsize / 72 * fig.dpi
        for bx0, by0, ha, va, dx, dy in (
            (px + gap, py - lh / 2, "left", "center", gap, 0),        # right
            (px - gap - w, py - lh / 2, "right", "center", -gap, 0),  # left
            (px - w / 2, py + gap, "center", "bottom", 0, gap),       # above
            (px - w / 2, py - gap - lh, "center", "top", 0, -gap),    # below
        ):
            b = (bx0, by0, bx0 + w, by0 + lh)
            if b[0] < ax0 or b[2] > ax1 or b[1] < ay0 or b[3] > ay1 or collides(b):
                continue
            for c in cells(b):
                grid.setdefault(c, []).append(b)
            ax.annotate(s, (float(x[i]), float(y[i])),
                        xytext=(dx * 72 / fig.dpi, dy * 72 / fig.dpi),
                        textcoords="offset points", fontsize=fontsize,
                        color=INK, ha=ha, va=va)
            n += 1
            break
    return n
