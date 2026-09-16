"""Is L0 attention winner-takes-all or distributed, per head?

For each head h and query position i, the softmax weights p over keys j <= i give
    p_max = max_j p_j          (1 = pure winner-takes-all)
    n_eff = exp(-sum p log p)  (effective number of attended keys; 1 = WTA,
                                i+1 = uniform)
computed over Pile rows (the model runs flash attention, so per-head weights are
recomputed from q/k with the block's own projections and RoPE buffers).

Writes ../head_concentration.png; histogram stats over queries i >= 64.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from load import load_pile_4l

N_ROWS, BATCH, T, QMIN = 300, 25, 512, 64

model, _, _ = load_pile_4l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

rows = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")
tokens = torch.stack([r[:T] for r in rows[:N_ROWS]]).to(device)

attn = model.h[0].attn
p_max_all, n_eff_all = [], []  # each (rows, 6, T)
with torch.no_grad():
    for b in range(0, N_ROWS, BATCH):
        x = model.h[0].rms_1(model.wte(tokens[b:b + BATCH]))
        B = x.shape[0]
        q = attn.q_proj(x).view(B, T, 6, 128).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, 6, 128).transpose(1, 2)
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos = attn.rotary_cos[pos].to(q.dtype)
        sin = attn.rotary_sin[pos].to(q.dtype)
        q, k = attn._apply_rotary_pos_emb(q, k, cos, sin)
        logits = q @ k.transpose(-2, -1) / np.sqrt(128)          # (B, 6, T, T)
        mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        p = logits.float().softmax(dim=-1)
        p_max_all.append(p.max(dim=-1).values.cpu())
        ent = -(p * p.clamp_min(1e-30).log()).sum(dim=-1)
        n_eff_all.append(ent.exp().cpu())

p_max = torch.cat(p_max_all).numpy()  # (rows, 6, T)
n_eff = torch.cat(n_eff_all).numpy()

print(f"{N_ROWS} rows, queries i >= {QMIN} ({N_ROWS * (T - QMIN)} per head)")
print("head | median p_max | p_max>0.5 | p_max>0.9 | median n_eff | n_eff q10..q90")
pm, ne = p_max[:, :, QMIN:], n_eff[:, :, QMIN:]
for h in range(6):
    a, e = pm[:, h].ravel(), ne[:, h].ravel()
    print(f"  h{h} |        {np.median(a):.3f} |    {(a > .5).mean():5.1%} |    {(a > .9).mean():5.1%} "
          f"|        {np.median(e):6.1f} | {np.quantile(e, .1):6.1f}..{np.quantile(e, .9):6.1f}")

colors = plt.cm.tab10(np.arange(6))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for h in range(6):
    axes[0].hist(pm[:, h].ravel(), bins=np.linspace(0, 1, 101), histtype="step",
                 density=True, color=colors[h], label=f"h{h}")
axes[0].set_xlabel("$p_{max}$ (top attention weight)")
axes[0].set_ylabel("density")
axes[0].set_title(f"Top weight, queries $i \\geq {QMIN}$")
axes[0].legend(fontsize=8)

bins = np.geomspace(1, T, 101)
for h in range(6):
    axes[1].hist(ne[:, h].ravel(), bins=bins, histtype="step",
                 density=True, color=colors[h], label=f"h{h}")
axes[1].set_xscale("log")
axes[1].set_xlabel("$n_{eff} = e^{H}$ (effective # keys)")
axes[1].set_title(f"Effective attended keys, queries $i \\geq {QMIN}$")

for h in range(6):
    axes[2].plot(np.median(n_eff[:, h], axis=0), color=colors[h], lw=1.2, label=f"h{h}")
axes[2].plot(np.arange(T) + 1, "k--", lw=0.8, label="uniform ($i+1$)")
axes[2].set_yscale("log")
axes[2].set_xlabel("query position $i$")
axes[2].set_ylabel("median $n_{eff}$")
axes[2].set_title("Effective keys vs query position")
axes[2].legend(fontsize=8)

fig.suptitle("L0 attention concentration per head (300 Pile rows)", fontsize=13)
fig.tight_layout()
out = HERE.parent / "head_concentration.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
