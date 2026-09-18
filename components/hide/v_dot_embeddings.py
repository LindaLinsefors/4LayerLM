"""Dot product of a C-decomposition component's read-in V with every token embedding.

Component: h.0.attn.q_proj:282 of decomposition C (p-d60af588, target t-87f91319, sink seed 45).
V from compare-decomps/hide/cache/uv_C.npz (all comps, id order), sign-fixed to the
majority-positive-activation gauge (act_signs_C.npz). Dot is taken with the RMSNorm-ed
embedding the component actually receives at L0: x_t = g * wte[t]/rms(wte[t]) (rms_1 gain g;
L0 attn input is token-only) — i.e. the x-axis is the component's true input activation.
Token frequencies = counts over the 4,000 cached Pile rows (2.052M tokens).
The 304 missing tokens are excluded from both histograms (project default).

Output: ../v_dot_embeddings/h.0.attn.q_proj/0.99-CI-282.png + outlier table on stdout.
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
from labeling import label_order, add_hist_labels

MOD, COMP = "h.0.attn.q_proj", 282
OUT_DIR = HERE.parent / "C" / "v_dot_embeddings" / MOD

# --- component read-in vector, sign-fixed ---
uv = np.load(ROOT / "compare-decomps/hide/cache/uv_C.npz")
V = uv[f"{MOD}|V"].astype(np.float32)  # (C, d_in): rows are components!
v = V[COMP]

s = np.load(ROOT / "compare-decomps/hide/cache/act_signs_C.npz")
Npos, F = int(s[f"{MOD}|Npos"][COMP]), int(s[f"{MOD}|F"][COMP])
Ssum, Sall = float(s[f"{MOD}|Ssum"][COMP]), float(s[f"{MOD}|Sall"][COMP])
if F > 0:
    sign = 1.0 if (Npos * 2 > F or (Npos * 2 == F and Ssum >= 0)) else -1.0
else:
    sign = 1.0 if Sall >= 0 else -1.0
v = sign * v

mean_ci = float(np.load(ROOT / "compare-decomps/hide/cache/coci_C.npz")[f"{MOD}|mean"][COMP])
print(f"{MOD}:{COMP}  sample mean CI = {mean_ci:.4f}  fires = {F}  sign gauge = {sign:+.0f}")

# --- embeddings & dots ---
with safe_open(ROOT / "sink-models/pretrain_cache/spd-t-87f91319/model_step_100000.safetensors",
               framework="np") as f:
    wte = f.get_tensor("wte.weight").astype(np.float32)  # (50277, 768)
    g = f.get_tensor("h.0.rms_1.weight").astype(np.float32)
rms = np.linalg.norm(wte, axis=1) / np.sqrt(wte.shape[1])
dots = ((wte / rms[:, None]) * g) @ v  # actual q_proj input activation

# --- token frequencies over the 4,000 cached Pile rows; missing tokens excluded ---
rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
counts = torch.bincount(torch.cat(list(rows)), minlength=wte.shape[0]).numpy()
missing = np.load(ROOT / "token-embeds/hide/missing_tokens.npy")
keep = np.ones(wte.shape[0], dtype=bool)
keep[missing] = False
d, c = dots[keep], counts[keep]
print(f"tokens: {keep.sum()} non-missing ({len(missing)} missing excluded), "
      f"{c.sum():,} corpus tokens")

# --- top firing tokens for context ---
tok = load_tokenizer("pile_4l")
tt = np.load(ROOT / "mean-ci-widget/hide/cache/top_tokens_C.npz")
ids, ci = tt[f"{MOD}|top_ids"][COMP], tt[f"{MOD}|top_ci"][COMP]
tot = float(tt[f"{MOD}|total"][COMP])
print("\nTop firing tokens (share of summed CI):")
for i, s_ci in zip(ids[:8], ci[:8]):
    if s_ci <= 0:
        break
    print(f"  {tok.decode([int(i)])!r:20s} {s_ci / tot:6.1%}   dot = {dots[int(i)]:+7.2f}")

# --- outliers ---
order = np.argsort(dots)
print("\nMost negative dots:                     Most positive dots:")
for a, b in zip(order[:12], order[::-1][:12]):
    print(f"  {tok.decode([int(a)])!r:24s} {dots[a]:+7.2f} n={counts[a]:<7d}"
          f"  {tok.decode([int(b)])!r:24s} {dots[b]:+7.2f} n={counts[b]:<6d}")

# --- figure ---
BLUE, INK = "#3D6DE2", "#39485E"
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for ax, w, title, ylab in [
    (axes[0], None, "each token weighted equally", "tokens per bin"),
    (axes[1], c, "weighted by corpus frequency", "corpus tokens per bin"),
]:
    ax.hist(d, bins=200, weights=w, color=BLUE, edgecolor="none")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[1].set_xlabel(r"$V_c \cdot \mathrm{rms}_1(\mathrm{wte}[t])$")

fig.suptitle(f"C  {MOD}:{COMP} — read-in $V_c$ · RMSNorm-ed token embeddings "
             r"($x_t = g \odot \mathrm{wte}[t]/\mathrm{rms}$; missing tokens excluded)",
             fontsize=12, color=INK)
fig.tight_layout()

# as many non-overlapping token labels as fit (labeling.py)
fig.canvas.draw()
def _name(i):
    t = tok.decode([int(i)])
    return (t.strip() or repr(t)).replace("$", r"\$")
names = [_name(i) for i in np.nonzero(keep)[0]]
lorder = label_order(d, c)
add_hist_labels(axes[0], fig, d, names, lorder)
add_hist_labels(axes[1], fig, d, names, lorder[c[lorder] > 0], weights=c)
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "0.99-CI-282.png", dpi=150)
print(f"\nsaved {OUT_DIR / '0.99-CI-282.png'}")
