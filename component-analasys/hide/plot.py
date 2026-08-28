"""Histogram of subcomponent magnitudes ||V_c|| * ||U_c|| (Pile 4-layer),
split by activity category. Reads cache/magnitudes.npz (built by compute.py).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
data = np.load(HERE / "cache" / "magnitudes.npz")
mag, cat = data["magnitude"], data["category"]

# (category code, label, color) -- codes from compute.py
GROUPS = [
    (0, "alive (mean CI > 1e-6)", "#2a78d6"),
    (1, "ever active, not alive", "#1baf7a"),
    (2, "never active", "#eda100"),
]

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.geomspace(mag.min(), mag.max(), 80)
for code, label, color in GROUPS:
    m = mag[cat == code]
    ax.hist(m, bins=bins, density=True, alpha=0.3, color=color,
            label=f"{label}  (n={len(m):,})")
    ax.hist(m, bins=bins, density=True, histtype="step", color=color, lw=1.5)

ax.set_xscale("log")
ax.set_xlabel(r"$\|V_c\| \, \|U_c\|$")
ax.set_ylabel("probability density")
ax.set_title("Pile 4L subcomponent magnitudes by activity")
ax.legend(frameon=False)
ax.grid(True, color="#e1e0d9", lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(HERE.parent / "magnitudes_hist.png", dpi=150)
print("saved", HERE / "magnitudes_hist.png")
