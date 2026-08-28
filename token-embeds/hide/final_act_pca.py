"""PCA of the final-residual-stream activations (after ln_f, just before the
unembedding) harvested by final_acts.py. Analog of pca.py for the embeddings,
without the token-group / frequency splits.

Top panel: variance along each PC (eigenvalue spectrum), both models, PCs from
the mean-centered activations. Bottom panels (one per model): mean projection
of the RAW (non-centered) activations onto each PC — i.e. mu . v_i, showing
which PCs carry the shared offset (the bias-direction analog); sign of each PC
fixed so the mean projection is >= 0.

Output: token-embeds/final_act_pca.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent / "final_act_pca.png"

MODEL_COLORS = {"pile_4l": "#4e79a7", "simple_2l": "#e0793d"}

fig, ((ax_spec, ax_spec2), (ax_p, ax_s)) = plt.subplots(2, 2, figsize=(12, 8.5))
ax_spec2.remove()
ax_spec.set_position([0.08, 0.56, 0.87, 0.38])  # spectrum spans the full top row
mean_axes = {"pile_4l": ax_p, "simple_2l": ax_s}

for name in ["pile_4l", "simple_2l"]:
    acts = np.load(HERE / "cache" / f"final_acts_{name}.npz")["acts"].astype(np.float64)
    n, d = acts.shape
    mu = acts.mean(axis=0)
    x = acts - mu
    cov = (x.T @ x) / n
    var, v = np.linalg.eigh(cov)          # ascending
    var, v = var[::-1], v[:, ::-1]        # descending: PC1 first

    ax_spec.plot(np.arange(1, d + 1), var, color=MODEL_COLORS[name], linewidth=2)
    ax_spec.annotate(
        f"{name}  ({n:,} positions, d={d})",
        xy=(d, var[-1]),
        xytext=(6, 0) if name == "simple_2l" else (-4, 8),
        textcoords="offset points",
        ha="left" if name == "simple_2l" else "right",
        va="center" if name == "simple_2l" else "bottom",
        fontsize=9, color=MODEL_COLORS[name],
    )
    top = ", ".join(f"{val:.3f}" for val in var[:3])
    print(f"{name}: total var {var.sum():.3f}, top PCs [{top}], "
          f"PCs for 50%/90% of var: {np.searchsorted(np.cumsum(var) / var.sum(), [0.5, 0.9]) + 1}")

    # mean projection onto each PC: raw activations, no centering = mu . v_i
    mean_proj = mu @ v
    mean_proj = np.abs(mean_proj)  # sign convention: mean >= 0
    ax = mean_axes[name]
    ax.plot(np.arange(1, d + 1), mean_proj, color=MODEL_COLORS[name], linewidth=1.2)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("principal component index")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

ax_spec.set_xlabel("principal component index")
ax_spec.set_ylabel("variance along PC (eigenvalue)")
ax_spec.set_yscale("log")
ax_spec.set_title("PCA spectrum of final-residual-stream (post-ln_f) activations on training data")
ax_spec.grid(alpha=0.25, linewidth=0.5)
ax_spec.spines[["top", "right"]].set_visible(False)
ax_p.set_ylabel("mean projection of raw activations onto PC\n(= μ·v, sign: mean ≥ 0)")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
