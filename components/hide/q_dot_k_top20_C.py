"""RoPE-rotated q.k score heatmaps for C, h.<l>.attn layers 0-3 -- top-20 zoom.

Same figures as q_dot_k_C.py (see its docstring for the score definition and
data sources) except: the y axis holds only the TOP-20 sample-mean-CI alive
components of the varying matrix (instead of all alive comps), and the x axis
only distances D = 0..99. Larger pixels per cell, every row labeled.

Output: components/C/q_dot_k_top20/<fixed matrix short>/<fixed CI:.2f>-CI-<id>.png
(folder names the FIXED component's matrix, standard components/C convention).
Local, no GPU, ~1 min per layer (160 figures total).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

N_HEAD, HEAD_DIM = 6, 128
N_D = 100
N_TOP = 20
DPI = 100
SCALE = 10

uv = np.load(ROOT / "compare-decomps/hide/cache/uv_C.npz")
cc = np.load(ROOT / "coci-heatmaps/hide/cache/coci_C.npz")
typ = np.load(HERE / "cache/typical_act_C.npz")

# corrected RoPE spectrum (rotate-half: plane p pairs dims p and p+64)
inv_freq = np.exp(np.load(
    ROOT / "sink-models/hide/cache/fitted_freqs_avg.npz")["log_inv_freq"])
ang = np.arange(N_D)[:, None] * inv_freq[None, :]          # (100, 64)
COS = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)   # (100, 128)
SIN = np.concatenate([np.sin(ang), np.sin(ang)], axis=1)


def rot(t):
    n = t.shape[-1] // 2
    return np.concatenate([-t[..., n:], t[..., :n]], axis=-1)


def scores(fixed_mod, fixed_id, vary_mod, order):
    """(N, 6, N_D) per-head score maps, typical-activation weighted."""
    f = uv[f"{fixed_mod}|U"][fixed_id].astype(np.float64).reshape(N_HEAD, -1)
    V = uv[f"{vary_mod}|U"][order].astype(np.float64).reshape(-1, N_HEAD,
                                                              HEAD_DIM)
    # rotate the fixed vector: +D if it is the query, -D if it is the key
    # (score(D) = (R_D q).k = q.(R_-D k))
    sgn = 1.0 if "q_proj" in fixed_mod else -1.0
    frot = f[None] * COS[:, None, :] + rot(f)[None] * (sgn * SIN)[:, None, :]
    S = np.einsum("dhp,nhp->nhd", frot, V) / np.sqrt(HEAD_DIM)
    t_f = typ[f"{fixed_mod}|typical"][fixed_id]
    t_v = typ[f"{vary_mod}|typical"][order]
    return S * t_f * t_v[:, None, None]


def make_figure(fixed_mod, fixed_id, vary_mod, order, out):
    S = scores(fixed_mod, fixed_id, vary_mod, order)
    total = S.sum(axis=1)
    scales = [("full max", np.abs(total).max(), np.abs(S).max()),
              ("fixed", 4.0, 2.0)]

    N = len(order)
    PW, PH = N_D * SCALE, N * SCALE
    ML, MGAP, MR = 64, 136, 120
    MT, TT, MB = 56, 40, 36
    CBGAP, CBW = 8, 14
    FW = ML + PW + MGAP + PW + MR
    ROW_H = TT + PH + MB
    FH = MT + 7 * ROW_H
    fig = plt.figure(figsize=(FW / DPI, FH / DPI), dpi=DPI)

    def rect(px, py, w, h):  # pixel rect (from top-left) -> figure fraction
        return [px / FW, 1 - (py + h) / FH, w / FW, h / FH]

    ytick = np.arange(N)
    maps = [total] + [S[:, h] for h in range(N_HEAD)]
    titles = ["all heads (sum)"] + [f"head {h}" for h in range(N_HEAD)]
    for r, (M, title) in enumerate(zip(maps, titles)):
        for c, (sname, vt, vh) in enumerate(scales):
            x0 = ML + c * (PW + MGAP)
            y0 = MT + r * ROW_H + TT
            ax = fig.add_axes(rect(x0, y0, PW, PH))
            vmax = vt if r == 0 else vh
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax,
                           vmax=vmax, interpolation="nearest")
            ax.set_title(f"{title}   (color range ±{vmax:.3g} logits, "
                         f"{sname}; this panel's |max| {np.abs(M).max():.3f})",
                         fontsize=9)
            ax.set_xlabel("distance D (key D tokens before query)", fontsize=8)
            ax.set_ylabel(f"top-{N_TOP} {vary_mod} comps, mean CI desc",
                          fontsize=8)
            ax.set_yticks(ytick)
            ax.set_yticklabels([str(order[i]) for i in ytick], fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
            # spines sit centered on the axes edge and would cover the edge
            # data rows -- move them out far enough to leave a visible white
            # gap, so a dark edge row cannot merge with the black frame
            for sp in ax.spines.values():
                sp.set_position(("outward", 2.0))
            cax = fig.add_axes(rect(x0 + PW + CBGAP, y0, CBW, PH))
            fig.colorbar(im, cax=cax)
            cax.tick_params(labelsize=7)

    fig.suptitle(
        f"C  {fixed_mod}:{fixed_id} · RoPE(D) · {vary_mod} (top-{N_TOP} by "
        f"mean CI), D 0..{N_D - 1}, weighted by typical activations\n"
        f"score = t_q t_k (R_D U_q)·U_k/√128 per head (logits at typical "
        "activation; t = CI- and corpus-weighted mean of x·V), corrected "
        "fitted RoPE spectrum; y ticks = component ids",
        fontsize=11, y=1 - 6 / FH, va="top")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


for fixed_mod, vary_mod in [(f"h.{l}.attn.{a}_proj", f"h.{l}.attn.{b}_proj")
                            for l in range(4) for a, b in
                            [("q", "k"), ("k", "q")]]:
    mean_f = cc[f"{fixed_mod}|mean"]
    mean_v = cc[f"{vary_mod}|mean"]
    order = np.argsort(mean_v)[::-1][:N_TOP]  # top-20 by mean CI desc
    top = np.argsort(mean_f)[::-1][:N_TOP]
    outdir = ROOT / "components/C/q_dot_k_top20" / fixed_mod.replace("_proj", "")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"fixed {fixed_mod} top-{N_TOP} vs top-{N_TOP} {vary_mod}:")
    for fid in top:
        out = outdir / f"{mean_f[fid]:.2f}-CI-{fid}.png"
        make_figure(fixed_mod, int(fid), vary_mod, order, out)
        print(f"  {fixed_mod}:{fid} (mean CI {mean_f[fid]:.3g}) -> {out.name}")
