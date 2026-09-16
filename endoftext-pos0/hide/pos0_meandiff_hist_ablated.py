"""pos0_meandiff_hist.py replicated with h.0.attn.o_proj ablated down to
components {389, 675, 1006} (all other 1021 rank-one matrices subtracted from
the weight, pos0_components.py convention) — does the pos-0 mean-difference
separation in the residual stream survive when attention-1's output can only
be written through the always-on component (389) and the two strongest pos-0
CI specialists (675, 1006)?

Same 1000 rows / bulk positions as the pos0_probe cache (rng seed 0), same
four stages, Delta recomputed from the ablated activations. Baseline ||Delta||
and AUC (from pos0_probe_acts.npz, unablated) are shown in each title.

Output: endoftext-pos0/pos0_meandiff_hist_ablated.png
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
N_BULK = 6
BULK_MIN = 128
MOD = "h.0.attn.o_proj"
KEEP = [389, 675, 1006]
STAGES = ["after attention 1", "after MLP 1", "after attention 2", "after MLP 2"]

# ------------------------------------------------------------ ablated forward
rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]
rng = np.random.default_rng(0)
bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))

model, pc, _ = load_pile_4l()
V, U = pc.components[MOD].V, pc.components[MOD].U
abl = np.setdiff1d(np.arange(V.shape[1]), KEEP)
lin = model.h[0].attn.o_proj
lin.weight.data -= (V[:, abl] @ U[abl, :]).T.to(lin.weight.dtype)
model = model.to(DEVICE)

caps: dict[str, torch.Tensor] = {}
hooks = [model.h[l].attn.register_forward_hook(
             lambda _m, _i, out, l=l: caps.__setitem__(f"attn{l}", out))
         for l in (0, 1)]
hooks += [model.h[l].register_forward_hook(
              lambda _m, _i, out, l=l: caps.__setitem__(f"block{l}", out))
          for l in (0, 1)]

h0, hb = [[] for _ in STAGES], [[] for _ in STAGES]
with torch.no_grad():
    for start in range(0, N_ROWS, BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        hs = [model.wte(batch) + caps["attn0"], caps["block0"]]
        hs += [hs[1] + caps["attn1"], caps["block1"]]
        tok = batch.cpu().numpy()
        for bi in range(len(tok)):
            ri = start + bi
            for si, h in enumerate(hs):
                hcpu = h[bi].float().cpu().numpy()
                if tok[bi, 0] != EOS_ID:
                    h0[si].append(hcpu[0])
                for p in bulk_pos[ri]:
                    if tok[bi, p] != EOS_ID:
                        hb[si].append(hcpu[p])
for h in hooks:
    h.remove()
del model

# --------------------------------------------- baseline stats from the cache
d = np.load(HERE / "cache" / "pos0_probe_acts.npz")
stage_names = [str(s) for s in d["stages"]]
acts, kept_tok = d["acts"], d["kept_tok"]
bulk_slots = slice(kept_tok.shape[1] - 6, None)
ok0 = kept_tok[:, 0] != EOS_ID
okb = kept_tok[:, bulk_slots] != EOS_ID


def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


def meandiff_proj(X0, Xb):
    delta = X0.mean(0) - Xb.mean(0)
    dn = np.linalg.norm(delta)
    return X0 @ (delta / dn), Xb @ (delta / dn), dn


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for si, (ax, stage) in enumerate(zip(axes.flat, STAGES)):
    X0 = np.array(h0[si], dtype=np.float64)
    Xb = np.array(hb[si], dtype=np.float64)
    x0, xb, dn = meandiff_proj(X0, Xb)

    ci = stage_names.index(stage)
    b0, bb, bdn = meandiff_proj(acts[ci, ok0, 0].astype(np.float64),
                                acts[ci][:, bulk_slots][okb].astype(np.float64))

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
                 f"pos-0 kurt {kurt(x0):.2f}{clip}\n"
                 f"(baseline: ‖Δ‖ = {bdn:.2f},  lin AUC {auc(bb, b0):.3f})",
                 fontsize=10.5)
    ax.set_xlabel("Δ̂·h,  Δ̂ ∝ mean(pos 0) − mean(bulk)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("pile_4l — mean-difference-direction histograms with h.0.attn.o_proj "
             "ablated to components {389, 675, 1006}\n(all other 1021 rank-one "
             "matrices subtracted; same 1000 rows as pos0_probe; each stage uses "
             "its own Δ̂ from the ablated run)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_meandiff_hist_ablated.png", dpi=150,
            bbox_inches="tight")
print(f"saved {HERE.parent / 'pos0_meandiff_hist_ablated.png'}")
