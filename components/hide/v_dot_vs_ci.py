"""Scatter: x = V_c · rms_1(wte[t]) (the component's actual L0 input activation:
g * wte[t]/rms(wte[t]), majority-positive gauge), y = mean CI of C
h.0.attn.q_proj:282 on token t, over the 4,000 cached Pile rows.

One point per token id seen at least once in the sample (missing tokens excluded;
count-0 tokens have no CI estimate). Color = log10 corpus count (mean-CI noise
shrinks with count). Requires cache/ci_per_token_282.npz from ci_per_token_modal.py.

⚠ CI computed through the public JAX sink loader = broken RoPE (rope_report.md);
token-identity-level statistics expected to survive qualitatively.

Output: ../v_dot_vs_ci/h.0.attn.q_proj/0.99-CI-282.png + low-CI outlier table.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from load import load_tokenizer
from labeling import add_scatter_labels

MOD, COMP = "h.0.attn.q_proj", 282
OUT_DIR = HERE.parent / "C" / "v_dot_vs_ci" / MOD

# --- dots (same recipe as v_dot_embeddings.py; gauge sign known to be -1) ---
uv = np.load(ROOT / "compare-decomps/hide/cache/uv_C.npz")
v = -uv[f"{MOD}|V"].astype(np.float32)[COMP]  # (C, d_in): row = component
with safe_open(ROOT / "sink-models/pretrain_cache/spd-t-87f91319/model_step_100000.safetensors",
               framework="np") as f:
    wte = f.get_tensor("wte.weight").astype(np.float32)
    g = f.get_tensor("h.0.rms_1.weight").astype(np.float32)
rms = np.linalg.norm(wte, axis=1) / np.sqrt(wte.shape[1])
dots = ((wte / rms[:, None]) * g) @ v  # actual q_proj input activation

# --- per-token mean CI ---
z = np.load(HERE / "cache" / "ci_per_token_282.npz")
rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
counts = torch.bincount(torch.cat(list(rows)), minlength=len(wte)).numpy()
missing = np.load(ROOT / "token-embeds/hide/missing_tokens.npy")
keep = counts > 0
keep[missing] = False
mean_ci = np.zeros(len(wte))
mean_ci[keep] = z["ci_sum"][keep] / counts[keep]
print(f"{keep.sum()} tokens with count > 0 (non-missing); "
      f"overall mean CI = {z['ci_sum'].sum() / counts.sum():.4f}")

# --- scatter ---
INK = "#39485E"
fig, ax = plt.subplots(figsize=(8, 5.5))
order = np.argsort(counts[keep])          # draw high-count points on top
sc = ax.scatter(dots[keep][order], mean_ci[keep][order],
                c=np.log10(counts[keep][order]), cmap="viridis",
                s=8, alpha=0.5, linewidths=0)
fig.colorbar(sc, ax=ax, label=r"$\log_{10}$ corpus count")
ax.set_xlabel(r"$V_c \cdot \mathrm{rms}_1(\mathrm{wte}[t])$")
ax.set_ylabel("mean CI on token $t$")
ax.set_title(f"C  {MOD}:{COMP} — input activation vs per-token mean CI\n"
             r"($x_t = g \odot \mathrm{wte}[t]/\mathrm{rms}$; 4,000 Pile rows; "
             "missing tokens excluded)", fontsize=11, color=INK)
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# stdout: the lowest-CI tokens with non-trivial counts
tok = load_tokenizer("pile_4l")
low = np.nonzero(keep & (counts >= 20))[0]
low = low[np.argsort(mean_ci[low])][:12]
print("\nLowest mean-CI tokens (count >= 20):")
for i in low:
    print(f"  {tok.decode([int(i)])!r:24s} CI = {mean_ci[i]:.3f}  "
          f"dot = {dots[i]:+.3f}  n = {counts[i]}")

fig.tight_layout()

# as many non-overlapping token labels as fit (labeling.py)
fig.canvas.draw()
def _name(i):
    t = tok.decode([int(i)])
    return (t.strip() or repr(t)).replace("$", r"\$")
names = [_name(i) for i in np.nonzero(keep)[0]]
ns = add_scatter_labels(ax, fig, dots[keep], mean_ci[keep], names, counts[keep])
print(f"\nscatter labels: {ns}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "0.99-CI-282.png", dpi=150)
print(f"\nsaved {OUT_DIR / '0.99-CI-282.png'}")
