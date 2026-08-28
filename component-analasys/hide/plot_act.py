"""Subcomponent activations a_c = ||U_c|| (V_c . phi) by activity category
(Pile 4-layer). Reads cache/activations.npz (built by compute_act.py) and
cache/magnitudes.npz (categories, built by compute.py).

Top left: per-component RMS of a_c over tokens -- the component's typical
interaction scale with the residual stream, no threshold needed.
Other panels: per-component fraction of tokens with |a_c| > theta ("how often
the component activates" under the paper's definition 2), one per threshold.

Both panels show the pdf per unit log10(x): with values spanning decades,
normalizing per linear x (plain density=True) would inflate the leftmost bins
by their tiny widths and visually bury everything else.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THETAS = [0.1, 0.3, 1.0]

HERE = Path(__file__).parent
act = np.load(HERE / "cache" / "activations.npz")
mag = np.load(HERE / "cache" / "magnitudes.npz")
assert (act["module"] == mag["module"]).all() and (act["index"] == mag["index"]).all()

cat = mag["category"]
n_tok = int(act["n_tokens"])
rms = np.sqrt(act["sum_sq"] / n_tok)


def exceedance_fraction(theta):
    first_bin = np.searchsorted(act["bin_edges"], theta) + 1  # first bin entirely > theta
    return act["hist"][:, first_bin:].sum(1) / n_tok

GROUPS = [
    (0, "alive (mean CI > 1e-6)", "#2a78d6"),
    (1, "ever active, not alive", "#1baf7a"),
    (2, "never active", "#eda100"),
]


def log_pdf_hist(ax, values, bins, color, label):
    """Overlapping histogram of dp/dlog10(x) on a log-scaled x axis."""
    d_log = np.log10(bins[1] / bins[0])
    weights = np.full(len(values), 1 / (len(values) * d_log))
    ax.hist(values, bins=bins, weights=weights, alpha=0.3, color=color, label=label)
    ax.hist(values, bins=bins, weights=weights, histtype="step", color=color, lw=1.5)


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax_rms = axes[0, 0]

bins_rms = np.geomspace(rms.min(), rms.max(), 80)
bins_frac = np.geomspace(1 / n_tok, 1, 60)
for code, label, color in GROUPS:
    log_pdf_hist(ax_rms, rms[cat == code], bins_rms, color,
                 f"{label}  (n={(cat == code).sum():,})")
ax_rms.set_xlabel(r"RMS of $a_c$ over tokens")
ax_rms.set_title("interaction scale with the residual stream")

for theta, ax in zip(THETAS, axes.flat[1:]):
    frac = exceedance_fraction(theta)
    for code, label, color in GROUPS:
        f = frac[cat == code]
        zero_pct = 100 * np.mean(f == 0)
        log_pdf_hist(ax, f[f > 0], bins_frac, color,
                     f"{label}  ({zero_pct:.0f}% at zero, excluded)")
    ax.set_xlabel(rf"fraction of tokens with $|a_c| > {theta}$")
    ax.set_title(f"how often the activation exceeds {theta}")

for ax in axes.flat:
    ax.set_xscale("log")
    ax.set_ylabel(r"probability density  $dp\,/\,d\log_{10}$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.suptitle(r"Pile 4L subcomponent activations  $a_c = \|U_c\|\,(V_c \cdot \varphi)$"
             "  (51,200 Pile tokens)")
fig.tight_layout()
fig.savefig(HERE.parent / "activations_hist.png", dpi=150)
print("saved", HERE / "activations_hist.png")
