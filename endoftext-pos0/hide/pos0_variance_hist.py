"""Histograms of the position-0 projections onto the top-3 variance-ratio
directions (from pos0_variance_signal.npz) plus the mean-difference direction
(mu_pos0 - mu_bulk, unit-normalized, computed from the same samples), layers 0
and 1 — a close-up of the distribution shapes (bimodal? spiky? shifted?) that
the summary kurtosis numbers compress. Bulk projections are overlaid for
reference (their peak is clipped: the y-limit is set by the pos-0 histogram).

Data: same 1000 rows / positions as pos0_variance_signal.py.
Output: endoftext-pos0/pos0_variance_hist.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 1000
BATCH_SIZE = 16
EOS_ID = 0
T = 512
N_BULK = 4
BULK_MIN = 128

cache = np.load(HERE / "cache" / "pos0_variance_signal.npz")
W = {l: cache[f"L{l}_W"][:, :3] for l in (0, 1)}

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]
rng = np.random.default_rng(0)
bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))

model, _, _ = load_pile_4l()
model = model.to(DEVICE)
caps: dict[int, torch.Tensor] = {}
hooks = [model.h[l].attn.o_proj.register_forward_pre_hook(
    lambda _m, inp, l=l: caps.__setitem__(l, inp[0])) for l in (0, 1)]

y0 = {l: [] for l in (0, 1)}
yb = {l: [] for l in (0, 1)}
with torch.no_grad():
    for start in range(0, N_ROWS, BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        tok = batch.cpu().numpy()
        for bi in range(len(tok)):
            ri = start + bi
            for l in (0, 1):
                y = caps[l][bi].float().cpu().numpy()
                if tok[bi, 0] != EOS_ID:
                    y0[l].append(y[0])
                for p in bulk_pos[ri]:
                    if tok[bi, p] != EOS_ID:
                        yb[l].append(y[p])
for h in hooks:
    h.remove()

# projections: top-3 variance directions + the mean-difference direction
p0, bulk = {}, {}
for l in (0, 1):
    Y0, Yb = np.array(y0[l], dtype=np.float64), np.array(yb[l], dtype=np.float64)
    dmean = Y0.mean(0) - Yb.mean(0)
    dmean /= np.linalg.norm(dmean)
    dirs = np.column_stack([W[l], dmean])
    p0[l], bulk[l] = Y0 @ dirs, Yb @ dirs


def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for l in (0, 1):
    for i in range(4):
        ax = axes[l, i]
        if i < 3:
            x0, xb = p0[l][:, i], bulk[l][:, i]
            name = f"dir {i}"
            xlabel = f"w·y (direction {i})"
        else:
            x0, xb = p0[l][:, 3], bulk[l][:, 3]
            name = "mean-diff dir"
            xlabel = "Δ̂·y,  Δ̂ ∝ μ_pos0 − μ_bulk"
        lo, hi = np.percentile(x0, [0.1, 99.9])
        pad = 0.05 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 50)
        dens0, _, _ = ax.hist(x0, bins=bins, density=True, color="tab:orange",
                              alpha=0.75, label=f"pos 0 (n={len(x0)})")
        ax.hist(xb, bins=bins, density=True, color="tab:blue", alpha=0.45,
                label="bulk (peak clipped)")
        ax.set_ylim(0, 1.25 * dens0.max())
        ratio = x0.var() / xb.var()
        ax.set_title(f"L{l} {name}: var ratio {ratio:.0f}, kurt {kurt(x0):.2f}, "
                     f"lin AUC {auc(xb, x0):.2f}", fontsize=10.5)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(f"layer {l}\ndensity")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("pile_4l — position-0 value-mixture projections onto the top-3 "
             "variance-ratio directions and the mean-difference direction\n"
             "(pre-o_proj y; variance directions from pos0_variance_signal.npz; "
             "two-point ±c code → kurtosis 1, Gaussian → 3)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_variance_hist.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'pos0_variance_hist.png'}")
