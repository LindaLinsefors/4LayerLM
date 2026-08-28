"""Plot NewGELU vs exact GELU vs ReLU, with an inset of the NewGELU-GELU difference."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

x = np.linspace(-4, 4, 801)

relu = np.maximum(x, 0)
gelu = x * 0.5 * (1 + erf(x / math.sqrt(2)))
new_gelu = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"  # Okabe-Ito, CVD-safe

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(x, relu, color=GREEN, lw=2, label="ReLU")
ax.plot(x, gelu, color=ORANGE, lw=2, label="GELU (exact)")
ax.plot(x, new_gelu, color=BLUE, lw=2, ls="--", label="NewGELU (tanh approx.)")

ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.axvline(0, color="0.85", lw=0.8, zorder=0)
ax.set_xlabel("x")
ax.set_ylabel("activation(x)")
ax.set_title("NewGELU vs exact GELU vs ReLU")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

inset = ax.inset_axes([0.62, 0.14, 0.34, 0.30])
inset.plot(x, 1e3 * (new_gelu - gelu), color=BLUE, lw=1.5)
inset.axhline(0, color="0.85", lw=0.8, zorder=0)
inset.set_title(r"NewGELU $-$ GELU  ($\times 10^3$)", fontsize=9)
inset.tick_params(labelsize=8)
inset.spines[["top", "right"]].set_visible(False)

out = Path(__file__).parent / "activation_functions.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved {out}")
print(f"max |NewGELU - GELU| = {np.abs(new_gelu - gelu).max():.2e} at x = {x[np.abs(new_gelu - gelu).argmax()]:.2f}")
