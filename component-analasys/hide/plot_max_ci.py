"""Histograms of the max causal importance ever reached per component, for the
alive and the ever-active-but-not-alive classes (Pile 4-layer). Reads
cache/max_ci_{alive,semi}.npz (built by compute_max_ci.py). Raw counts; the
bottom row zooms the y axis to make the sparse tails visible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ZOOM_YMAX = 50

HERE = Path(__file__).parent
CLASSES = [
    ("alive", "alive (mean CI > 1e-6)", "#2a78d6"),
    ("semi", "ever active, not alive", "#1baf7a"),
]

values = {cls: np.load(HERE / "cache" / f"max_ci_{cls}.npz")["max_ci"] for cls, _, _ in CLASSES}
bins = np.geomspace(min(v.min() for v in values.values()), 1, 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for (cls, label, color), (ax_full, ax_zoom) in zip(CLASSES, axes.T):
    for ax in (ax_full, ax_zoom):
        ax.hist(values[cls], bins=bins, alpha=0.3, color=color)
        ax.hist(values[cls], bins=bins, histtype="step", color=color, lw=1.5)
        ax.set_xscale("log")
        ax.set_ylabel("components per bin")
        ax.grid(True, color="#e1e0d9", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_full.set_title(f"{label}  (n={len(values[cls]):,})")
    ax_zoom.set_ylim(0, ZOOM_YMAX)
    ax_zoom.set_title(f"same, y axis zoomed to {ZOOM_YMAX}")
    ax_zoom.set_xlabel("max causal importance over harvested examples")

fig.suptitle("Pile 4L: max CI ever reached per component")
fig.tight_layout()
fig.savefig(HERE.parent / "max_ci_hist.png", dpi=150)
print("saved", HERE / "max_ci_hist.png")
