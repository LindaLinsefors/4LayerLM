"""RoPE-rotated q.k score heatmaps for decomposition F, h.<l>.attn, layers 0-3.

For each FIXED component (the top-20 sample-mean-CI comps of each layer's
q_proj and k_proj), heatmaps of the attention-score contribution against
every alive component of the OPPOSITE matrix as a function of query-key
distance D, per head and summed over heads:

    score_h(c, D) = t_fix t_c (R_D U_q,h) . U_k,c,h / sqrt(128)

U = the components' write (output) factors reshaped to (6 heads, 128); R_D is
the RoPE rotation at relative position D applied to the query side
(relative-position identity (R_i q).(R_j k) = (R_{i-j} q).k; for a fixed k
component we equivalently rotate it by -D and dot with every alive q comp).
t = TYPICAL ACTIVATION: the CI- and token-frequency-weighted mean of the input
activation a = x @ V over the 4,000 cached Pile rows, t = sum CI*a / sum CI,
from cache/typical_act_F.npz (typical_act_modal_F.py, fitted-RoPE forward).
The map is the attention-logit contribution of the pair when both components
sit at their typical activations; the product is gauge-invariant, so no
sign-gauge fixing is needed. RoPE uses the FITTED corrected frequencies
(sink-models/hide/cache/fitted_freqs_avg.npz, rotate-half convention, dims
(p, p+64) paired) -- for F this IS the training-time spectrum (F was trained
with it by sink-models/redo-decomps/), so unlike C there is no mis-load
caveat anywhere in this pipeline.

Per figure: 7 panel rows (all-heads sum + heads 0-5; head panels sum to the
total) x 2 columns (full-max color scale | fixed +-4 logits total / +-2 per
head); y = alive comps of the varying matrix by sample mean CI desc (ticks =
component ids), x = D 0..511. Exactly SCALE px per step/component (manual
pixel layout; spines offset outward so edge rows are not covered).

Output: components/F/q_dot_k/<fixed matrix short>/<fixed CI:.2f>-CI-<id>.png
(folder names the FIXED component's matrix, like the other components/F
figure types: h.<l>.attn.q holds that layer's top-20 q comps' figures, each vs
all alive k comps of the same layer, and vice versa). Local, no GPU, ~2 min
per layer (160 figures total).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

N_HEAD, HEAD_DIM = 6, 128
N_CTX = 512
N_TOP = 20
DPI = 100
SCALE = 2

uv = np.load(ROOT / "compare-decomps/hide/cache/uv_F.npz")
cc = np.load(ROOT / "coci-heatmaps/hide/cache/coci_F.npz")
typ = np.load(HERE / "cache/typical_act_F.npz")

# corrected RoPE spectrum (rotate-half: plane p pairs dims p and p+64)
inv_freq = np.exp(np.load(
    ROOT / "sink-models/hide/cache/fitted_freqs_avg.npz")["log_inv_freq"])
ang = np.arange(N_CTX)[:, None] * inv_freq[None, :]        # (512, 64)
COS = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)   # (512, 128)
SIN = np.concatenate([np.sin(ang), np.sin(ang)], axis=1)


def rot(t):
    n = t.shape[-1] // 2
    return np.concatenate([-t[..., n:], t[..., :n]], axis=-1)


def scores(fixed_mod, fixed_id, vary_mod, order):
    """(N, 6, 512) per-head score maps, typical-activation weighted."""
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
    PW, PH = N_CTX * SCALE, N * SCALE
    ML, MGAP, MR = 64, 136, 120
    MT, TT, MB = 56, 40, 36
    CBGAP, CBW = 8, 14
    FW = ML + PW + MGAP + PW + MR
    ROW_H = TT + PH + MB
    FH = MT + 7 * ROW_H
    fig = plt.figure(figsize=(FW / DPI, FH / DPI), dpi=DPI)

    def rect(px, py, w, h):  # pixel rect (from top-left) -> figure fraction
        return [px / FW, 1 - (py + h) / FH, w / FW, h / FH]

    ytick = np.arange(0, N, 20)
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
            ax.set_ylabel(f"alive {vary_mod} comps, mean CI desc", fontsize=8)
            ax.set_yticks(ytick)
            ax.set_yticklabels([str(order[i]) for i in ytick], fontsize=4.5)
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
        f"F  {fixed_mod}:{fixed_id} · RoPE(D) · {vary_mod} (alive, {N} "
        f"comps), weighted by typical activations\n"
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
    alive = np.flatnonzero(mean_v > 1e-6)
    order = alive[np.lexsort((alive, -mean_v[alive]))]  # CI desc, ties by id
    top = np.argsort(mean_f)[::-1][:N_TOP]
    outdir = ROOT / "components/F/q_dot_k" / fixed_mod.replace("_proj", "")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"fixed {fixed_mod} top-{N_TOP} vs {len(order)} alive {vary_mod}:")
    for fid in top:
        out = outdir / f"{mean_f[fid]:.2f}-CI-{fid}.png"
        make_figure(fixed_mod, int(fid), vary_mod, order, out)
        print(f"  {fixed_mod}:{fid} (mean CI {mean_f[fid]:.3g}) -> {out.name}")
