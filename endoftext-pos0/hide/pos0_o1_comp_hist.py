"""Histograms of the L0 o_proj input projections V_c.y (pre-o_proj value
mixture y, component read-in V_c) at position 0 vs bulk, in the style of
pos0_bimodal_dirs.png, for three component sets chosen from the per-position
CI stats (pos0_ci.npz, 4000 rows):

  rows 1-2: top-8 by mean CI at pos 0 minus overall mean CI (pos-0 specialists)
  row 3:    top-4 by overall mean CI
  row 4:    4 random components among those that ever fired (CI > 0.1), seed 0,
            excluding the 12 above

Data: same 1000 rows / bulk positions as pos0_variance_hist.py.
Output: endoftext-pos0/pos0_o1_comp_hist.png
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
MOD = "h.0.attn.o_proj"

# ------------------------------------------------- component selection from CI
d = np.load(HERE / "cache" / "pos0_ci.npz")
n_rows_ci = int(d["n_rows"])
Sp, Fp = d[f"{MOD}|Sp"], d[f"{MOD}|Fp"]
mean0 = Sp[:, 0] / n_rows_ci
overall = Sp.sum(1) / (n_rows_ci * T)
top_diff = np.argsort(-(mean0 - overall))[:8]
top_all = np.argsort(-overall)[:4]
rng = np.random.default_rng(0)
fired = np.where(Fp.sum(1) > 0)[0]
pool = np.setdiff1d(fired, np.concatenate([top_diff, top_all]))
rand4 = rng.choice(pool, 4, replace=False)

groups = [(list(top_diff[:4]), "top ΔCI"), (list(top_diff[4:]), "top ΔCI"),
          (list(top_all), "top overall CI"), (list(rand4), "random fired")]
print("top-8 diff:", list(top_diff), " top-4 overall:", list(top_all),
      " random:", list(rand4))

# ------------------------------------------------------------------- capture
rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]
bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))

model, pc, _ = load_pile_4l()
V = pc.components[MOD].V.float().numpy()           # (d_in, C); s_c = V[:, c].y
model = model.to(DEVICE)
caps: dict[int, torch.Tensor] = {}
hook = model.h[0].attn.o_proj.register_forward_pre_hook(
    lambda _m, inp: caps.__setitem__(0, inp[0]))

y0, yb = [], []
with torch.no_grad():
    for start in range(0, N_ROWS, BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        tok = batch.cpu().numpy()
        for bi in range(len(tok)):
            ri = start + bi
            y = caps[0][bi].float().cpu().numpy()
            if tok[bi, 0] != EOS_ID:
                y0.append(y[0])
            for p in bulk_pos[ri]:
                if tok[bi, p] != EOS_ID:
                    yb.append(y[p])
hook.remove()
Y0 = np.array(y0, dtype=np.float64)
Yb = np.array(yb, dtype=np.float64)
print(f"pos-0 samples: {len(Y0)}; bulk: {len(Yb)}")


def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


# ---------------------------------------------------------------------- plot
fig, axes = plt.subplots(4, 4, figsize=(19, 15))
for gi, (comps, gname) in enumerate(groups):
    for ci, c in enumerate(comps):
        ax = axes[gi, ci]
        x0, xb = Y0 @ V[:, c], Yb @ V[:, c]
        lo, hi = np.percentile(x0, [0.2, 99.8])
        pad = 0.05 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 45)
        dens0, _, _ = ax.hist(x0, bins=bins, density=True, color="tab:orange",
                              alpha=0.75, label=f"pos 0 (n={len(x0)})")
        ax.hist(xb, bins=bins, density=True, color="tab:blue", alpha=0.45,
                label="bulk (peak clipped)")
        ax.set_ylim(0, 1.25 * dens0.max())
        ax.set_title(f"{gname} — o1:{c}\nmean CI@0 {mean0[c]:.2f}, overall "
                     f"{overall[c]:.4f}, var ratio {x0.var() / xb.var():.1f},\n"
                     f"kurt {kurt(x0):.2f}, lin AUC {auc(xb, x0):.2f}",
                     fontsize=10)
        ax.set_xlabel("V_c·y")
        if ci == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("pile_4l — h.0.attn.o_proj component input projections V_c·y at "
             "position 0 vs bulk\n(pre-o_proj value mixture; component sets "
             "from pos0_ci.npz: top-8 mean CI@pos0 − overall, top-4 overall, "
             "4 random fired)", fontsize=13)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_o1_comp_hist.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'pos0_o1_comp_hist.png'}")
