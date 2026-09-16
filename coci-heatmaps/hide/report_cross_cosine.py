"""Cross-decomposition U- and V-cosine-similarity reports: old vs newA vs newB.

Same structure and EXACTLY the same axis orders as report_cross.py
(old-newA-newB/report_co_ci.md; the diagonal-matched ordering derived from
co-CI r via diagonal_orders(), NOT from the cosines — cells are directly
comparable across the three reports), but instead of co-CI each heatmap shows
the absolute cosine similarity between the rank-one factors of alive
components of two decompositions:

  U report: |cos(U_a, U_b)|, U in R^{d_out} — the component's write vector.
  V report: |cos(V_a, V_b)|, V in R^{d_in}  — the component's read-in vector.

Absolute value because a component's sign is gauge ((V_c, U_c) -> (-V_c, -U_c)
is the same rank-one matrix).

Inputs: cache/uv_{newA,newB}.npz (from uv_dump_new_modal.py),
cache/cross_partial_{newA,newB}.npz + cross_alive.npz and the mean-CI caches
(for the shared ordering). The old decomposition's factors are extracted here
directly from the local torch checkpoint on first run -> cache/uv_old.npz.

Writes old-newA-newB/report_u_cosine.md + report_v_cosine.md; figures in
hide/figures/cross_cos/{U,V}/<pair>/.

Usage: python coci-heatmaps/hide/report_cross_cosine.py   (~4 min)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE.parent / "old-newA-newB"
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from report_cross import (MODS, PAIRS, MATCH_R, cross_stats,  # noqa: E402
                          diagonal_orders, heatmap)
from report import MATS  # noqa: E402


def old_uv(alive):
    """Alive-component factors of the old decomposition, same npz layout as the
    Modal dumps: <mod>|V (n_alive, d_in), <mod>|U (n_alive, d_out), float16,
    rows in ascending alive-id order."""
    path = CACHE / "uv_old.npz"
    if not path.exists():
        import torch
        ckpt = (ROOT / "prev_paper" / "models" / "pile_4layer"
                / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")
        print("extracting old U/V from", ckpt.name, "...", flush=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        data = {}
        for mod in MODS:
            key = "_components." + mod.replace(".", "-")
            V = state[key + ".V"].float().numpy()   # (d_in, C)
            U = state[key + ".U"].float().numpy()   # (C, d_out)
            ids = alive[f"old|{mod}"]
            data[f"{mod}|V"] = V[:, ids].T.astype(np.float16)
            data[f"{mod}|U"] = U[ids].astype(np.float16)
        np.savez(path, **data)
    return np.load(path)


def unit(rows: np.ndarray) -> np.ndarray:
    x = rows.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = np.inf
    return x / n


INTRO = """# Cross-decomposition {m}-cosine similarity: old vs newA vs newB

Same structure as the co-CI report [report_co_ci.md](report_co_ci.md), but \
instead \
of co-CI each heatmap shows the **absolute cosine similarity \
|cos({m}_a, {m}_b)| between the {desc} of an alive component of one \
decomposition and one of another** — {m} is the {expl} of the rank-one \
subcomponent V_c U_c^T. Pairs (old, newA), (old, newB), (newA, newB); the \
first-named decomposition is the y axis. Absolute value because a component's \
sign is gauge ((V_c, U_c) -> (-V_c, -U_c) is the same component). Color \
white -> red spans 0 -> 1; the random baseline is E|cos| = sqrt(2/(pi d)) \
~= 0.03 for d = 768, 0.014 for d = 3072.

Alive components only, in **exactly the axis orders of \
[report_co_ci.md](report_co_ci.md)** (y: descending mean CI; x: each \
component at its best-co-CI-matching y component's position when that \
r >= {match_r}, unmatched at the right by mean CI — the matching comes from \
co-CI r, NOT from the cosines), so every cell is directly comparable across \
the three reports: a diagonal-band cell that is red both there and here is a \
matched mechanism whose factors also geometrically agree."""


def main() -> None:
    alive, stats, G, T = cross_stats()
    uv = {"old": old_uv(alive),
          "newA": np.load(CACHE / "uv_newA.npz"),
          "newB": np.load(CACHE / "uv_newB.npz")}
    orders = diagonal_orders(alive, stats, G, T)

    for meas, desc, expl in [
            ("U", "write vectors", "output (d_out) factor"),
            ("V", "read-in vectors", "input (d_in) factor")]:
        lines = [INTRO.format(m=meas, desc=desc, expl=expl,
                              match_r=MATCH_R), ""]
        prev_blk = None
        for l in range(4):
            for mat in MATS:
                mod = f"h.{l}.{mat}"
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
                    Y = unit(uv[ny_][f"{mod}|{meas}"])[row_pos]
                    X = unit(uv[nx_][f"{mod}|{meas}"])[col]
                    c = np.abs(Y @ X.T)
                    ids_y = alive[f"{ny_}|{mod}"][row_pos]
                    ids_x = alive[f"{nx_}|{mod}"][col]
                    pair = f"{ny_}_{nx_}"
                    fig_dir = HERE / "figures" / "cross_cos" / meas / pair
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"h{l}_{mat.replace('.', '_')}.png"
                    title = (f"{ny_} × {nx_}   {mod}\ncross |cos({meas})|, "
                             f"{c.shape[0]} × {c.shape[1]} alive components "
                             f"({n_matched} co-CI-matched at r ≥ {MATCH_R})")
                    heatmap(c, ids_y, ids_x, ny_, nx_, title,
                            fig_dir / fname, cmap="Reds", vmin=0, vmax=1,
                            note_y="mean CI order",
                            note_x="best co-CI-match position; "
                                   "unmatched right, by mean CI")
                    rel = f"../hide/figures/cross_cos/{meas}/{pair}/{fname}"
                    lines += [f"![{mod} {meas} {pair}]({rel})", ""]
                    print(f"{meas} {mod} {pair}: {c.shape[0]}x{c.shape[1]}, "
                          f"max {c.max():.3f}", flush=True)
        path = OUT / f"report_{meas.lower()}_cosine.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
