"""Histograms of RESIDUAL-STREAM activations projected onto the per-stage
mean-difference direction Delta = mean(pos 0) - mean(bulk), for the four
stages after attention 1 / MLP 1 / attention 2 / MLP 2 — how the position-0
separation (and the bimodal split seen in the pre-o_proj value mixture)
evolves through the residual stream.

Data: the pos0_probe.py cache (10 stages x 1000 rows x {15 early + 6 bulk
positions}); pos-0 class = position 0 (n~999), bulk class = the 6 random
positions >= 128 per row (n~6000); EOS tokens excluded. "Everywhere" is
represented by bulk: positions < 128 are a vanishing fraction of all tokens.

Output: endoftext-pos0/pos0_meandiff_hist.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
EOS_ID = 0
STAGES = ["after attention 1", "after MLP 1", "after attention 2", "after MLP 2"]

d = np.load(HERE / "cache" / "pos0_probe_acts.npz")
stage_names = [str(s) for s in d["stages"]]
acts, kept_tok = d["acts"], d["kept_tok"]
n_early = kept_tok.shape[1] - 6
bulk_slots = slice(n_early, None)
ok0 = kept_tok[:, 0] != EOS_ID
okb = kept_tok[:, bulk_slots] != EOS_ID


def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, stage in zip(axes.flat, STAGES):
    si = stage_names.index(stage)
    X0 = acts[si, ok0, 0].astype(np.float64)
    Xb = acts[si][:, bulk_slots][okb].astype(np.float64)
    delta = X0.mean(0) - Xb.mean(0)
    dn = np.linalg.norm(delta)
    w = delta / dn
    x0, xb = X0 @ w, Xb @ w

    lo = min(np.percentile(x0, 0.2), np.percentile(xb, 0.2))
    hi = max(np.percentile(x0, 99.8), np.percentile(xb, 99.8))
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 60)
    d0, _, _ = ax.hist(x0, bins=bins, density=True, color="tab:orange", alpha=0.75,
                       label=f"pos 0 (n={len(x0)})")
    db, _, _ = ax.hist(xb, bins=bins, density=True, color="tab:blue", alpha=0.45,
                       label=f"bulk ≥ 128 (n={len(xb)})")
    if db.max() > 3 * d0.max():
        ax.set_ylim(0, 1.25 * d0.max())
        clip = ", bulk peak clipped"
    else:
        clip = ""
    ax.set_title(f"{stage}:  ‖Δ‖ = {dn:.2f},  lin AUC {auc(xb, x0):.3f},  "
                 f"pos-0 kurt {kurt(x0):.2f}{clip}", fontsize=10.5)
    ax.set_xlabel("Δ̂·h,  Δ̂ ∝ mean(pos 0) − mean(bulk)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("pile_4l — residual stream projected onto the per-stage mean-difference "
             "direction, position 0 vs bulk\n(1000 Pile rows from the pos0_probe "
             "cache; EOS excluded; each stage uses its own Δ̂)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_meandiff_hist.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'pos0_meandiff_hist.png'}")
