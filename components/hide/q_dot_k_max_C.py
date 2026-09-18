"""Max/min-over-distance RoPE-rotated q.k heatmaps for decomposition C, h.0.attn.

Per head h, the unweighted attention-score contribution of every alive q/k
component pair at query-key distance D:

    score_h(q, k, D) = (R_D U_q,h) . U_k,h / sqrt(128)

(U = write factors reshaped to (6 heads, 128); R_D = the FITTED corrected RoPE
spectrum -- sink-models/hide/cache/fitted_freqs_avg.npz, rotate-half, dims
(p, p+64) paired -- NOT the recorded-but-wrong rotary_base 10000, see
sink-models/rope_report.md). Component sign gauge is fixed by the sign of each
component's typical activation t (CI-weighted mean of x@V over the 4,000
cached Pile rows, cache/typical_act_C.npz), so signs are meaningful; unlike
the q_dot_k figures the scores are NOT multiplied by t_q t_k.

Two panels (max over D = 0..511 | min over D) for the all-heads sum (heads
summed per D, then reduced) and for each head separately. y = alive
h.0.attn.q_proj comps, x = alive h.0.attn.k_proj comps, both sorted by sample
mean CI descending (ties by id); one shared symmetric color scale per figure.

Outputs: components/C/q_dot_k_max/h.0.attn.png (all alive comps) and
h.0.attn_top20.png (top-20 mean-CI comps per side = the top-left corner of the
full maps, every tick id-labeled). Local, no GPU, ~1 min.
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

uv = np.load(ROOT / "compare-decomps/hide/cache/uv_C.npz")
cc = np.load(ROOT / "coci-heatmaps/hide/cache/coci_C.npz")
typ = np.load(HERE / "cache/typical_act_C.npz")

inv_freq = np.exp(np.load(
    ROOT / "sink-models/hide/cache/fitted_freqs_avg.npz")["log_inv_freq"])
ang = np.arange(N_CTX)[:, None] * inv_freq[None, :]        # (512, 64)
COS = np.concatenate([np.cos(ang), np.cos(ang)], axis=1).astype(np.float32)
SIN = np.concatenate([np.sin(ang), np.sin(ang)], axis=1).astype(np.float32)


def rot(t):
    n = t.shape[-1] // 2
    return np.concatenate([-t[..., n:], t[..., :n]], axis=-1)


def alive_order(mod):
    mean = cc[f"{mod}|mean"]
    alive = np.flatnonzero(mean > 1e-6)
    return alive[np.lexsort((alive, -mean[alive]))]  # CI desc, ties by id


def gauged_U(mod, order):
    """(N, 6, 128) write factors, sign-gauged by typical-activation sign."""
    U = uv[f"{mod}|U"][order].astype(np.float32)
    sgn = np.where(typ[f"{mod}|typical"][order] < 0, -1.0, 1.0)
    return (U * sgn[:, None]).reshape(-1, N_HEAD, HEAD_DIM)


QMOD, KMOD = "h.0.attn.q_proj", "h.0.attn.k_proj"
oq, ok = alive_order(QMOD), alive_order(KMOD)
Uq, Uk = gauged_U(QMOD, oq), gauged_U(KMOD, ok)
Nq, Nk = len(oq), len(ok)
print(f"alive: {Nq} q x {Nk} k comps")

# per head: rotate every q comp by every D, dot with every k comp; the
# max/min over D must be taken AFTER summing heads for the combined panels
# (max of sum != sum of maxes), so the all-heads score is accumulated per D
Smax = np.empty((N_HEAD + 1, Nq, Nk), np.float32)  # row 0 = all heads (sum)
Smin = np.empty((N_HEAD + 1, Nq, Nk), np.float32)
Stot = np.zeros((Nq, Nk, N_CTX), np.float32)
for h in range(N_HEAD):
    Q, K = Uq[:, h], Uk[:, h]
    frot = Q[None] * COS[:, None, :] + rot(Q)[None] * SIN[:, None, :]
    S = np.einsum("dqp,kp->qkd", frot, K) / np.sqrt(HEAD_DIM)  # (Nq, Nk, 512)
    Smax[h + 1], Smin[h + 1] = S.max(axis=-1), S.min(axis=-1)
    Stot += S
    print(f"head {h}: max {Smax[h + 1].max():.3f}, min {Smin[h + 1].min():.3f}")
Smax[0], Smin[0] = Stot.max(axis=-1), Stot.min(axis=-1)
del Stot, S
print(f"all heads: max {Smax[0].max():.3f}, min {Smin[0].min():.3f}")


def make_figure(Smax, Smin, oq, ok, scale, tick_step, out, subtitle):
    nq, nk = Smax.shape[1], Smax.shape[2]
    # separate symmetric scales: all-heads row | head rows (shared)
    vtot = max(np.abs(Smax[0]).max(), np.abs(Smin[0]).max())
    vhead = max(np.abs(Smax[1:]).max(), np.abs(Smin[1:]).max())
    PW, PH = nk * scale, nq * scale
    ML, MGAP, MR = 84, 150, 120
    MT, TT, MB = 64, 40, 56
    CBGAP, CBW = 8, 14
    FW = ML + PW + MGAP + PW + MR
    ROW_H = TT + PH + MB
    FH = MT + (N_HEAD + 1) * ROW_H
    fig = plt.figure(figsize=(FW / DPI, FH / DPI), dpi=DPI)

    def rect(px, py, w, h):  # pixel rect (from top-left) -> figure fraction
        return [px / FW, 1 - (py + h) / FH, w / FW, h / FH]

    yt, xt = np.arange(0, nq, tick_step), np.arange(0, nk, tick_step)
    tfs = 4.5 if tick_step > 1 else 7
    titles = ["all heads (sum)"] + [f"head {h}" for h in range(N_HEAD)]
    for r, title in enumerate(titles):
        for c, (M, rname) in enumerate([(Smax[r], "max"), (Smin[r], "min")]):
            x0 = ML + c * (PW + MGAP)
            y0 = MT + r * ROW_H + TT
            ax = fig.add_axes(rect(x0, y0, PW, PH))
            vmax = vtot if r == 0 else vhead
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax,
                           vmax=vmax, interpolation="nearest")
            ax.set_title(f"{title} — {rname} over D   (panel max "
                         f"{M.max():.3f}, min {M.min():.3f})", fontsize=9)
            ax.set_xlabel(f"alive {KMOD} comps, mean CI desc", fontsize=8)
            ax.set_ylabel(f"alive {QMOD} comps, mean CI desc", fontsize=8)
            ax.set_yticks(yt)
            ax.set_yticklabels([str(oq[i]) for i in yt], fontsize=tfs)
            ax.set_xticks(xt)
            ax.set_xticklabels([str(ok[i]) for i in xt], fontsize=tfs)
            ax.tick_params(axis="x", labelrotation=90)
            # spines offset outward so edge rows/cols stay visible
            for sp in ax.spines.values():
                sp.set_position(("outward", 2.0))
            cax = fig.add_axes(rect(x0 + PW + CBGAP, y0, CBW, PH))
            fig.colorbar(im, cax=cax)
            cax.tick_params(labelsize=7)

    fig.suptitle(
        f"C  h.0.attn: max/min over distance D=0..511 of "
        f"(R_D U_q)·U_k/√128 per head ({subtitle})\n"
        "unweighted; sign gauge = typical-activation sign; corrected fitted "
        f"RoPE spectrum; color range ±{vtot:.3f} all-heads / ±{vhead:.3f} "
        "head panels; ticks = component ids",
        fontsize=11, y=1 - 6 / FH, va="top")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


outdir = ROOT / "components/C/q_dot_k_max"
outdir.mkdir(parents=True, exist_ok=True)
make_figure(Smax, Smin, oq, ok, 2, 20, outdir / "h.0.attn.png",
            f"all {Nq} x {Nk} alive comps")
make_figure(Smax[:, :N_TOP, :N_TOP], Smin[:, :N_TOP, :N_TOP],
            oq[:N_TOP], ok[:N_TOP], 24, 1, outdir / "h.0.attn_top20.png",
            f"top-{N_TOP} mean-CI comps per side")
