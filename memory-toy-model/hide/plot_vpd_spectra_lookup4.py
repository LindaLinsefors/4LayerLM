"""Mean-CI spectra for the C=8192 > n_facts=4096 lookup-CI runs -> vpd_mean_ci_lookup4.png.

3x3 grid: rows = arch, cols = minimality coeff (1e-4 / 1e-5 / 1e-6). Log x (C spans
8192). Reads hide/cache/vpd_mean_ci_lookup4.npz (keys "<arch>|<coeff:g>|<site>").
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = np.load(HERE / "cache" / "vpd_mean_ci_lookup4.npz")

SITE_COLORS = {
    "q_proj": "#2a78d6", "k_proj": "#7db3e8", "v_proj": "#1baf7a",
    "o_proj": "#84d6b8", "mix": "#8a5cd6", "c_fc": "#d6552a", "down_proj": "#e8a37d",
}
ARCHS = ["attn", "twoemb", "mix"]
COEFFS = ["0.0001", "1e-05", "1e-06"]

fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True, sharex=True)
for i, arch in enumerate(ARCHS):
    for j, coeff in enumerate(COEFFS):
        ax = axes[i, j]
        sites = sorted({key.split("|")[2] for key in data.files
                        if key.startswith(f"{arch}|{coeff}|")})
        for site in sites:
            values = np.sort(data[f"{arch}|{coeff}|{site}"])[::-1]
            n_alive = int((values > 1e-6).sum())
            ax.plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-12),
                    color=SITE_COLORS[site], lw=1.4,
                    label=f"{site} (alive {n_alive})")
        ax.axhline(1e-6, color="green", ls="--", lw=0.8)
        ax.axvline(4096, color="gray", ls=":", lw=1)  # n_facts
        ax.axvline(96, color="gray", ls="--", lw=0.8)  # matrix rank
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 2)
        ax.set_title(f"{arch}, coeff {coeff}", fontsize=11)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25)
        if i == 2:
            ax.set_xlabel("component rank (by mean CI, desc; log)")
        if j == 0:
            ax.set_ylabel("mean CI (lower_leaky, all facts)")
fig.suptitle("C=8192 > n_facts=4096 (k12 targets), lookup-table CI: mean-CI spectra.\n"
             "Vertical lines: matrix rank 96 (dashed), n_facts 4096 (dotted); C = 8192 per matrix",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE.parent / "vpd_mean_ci_lookup4.png"
fig.savefig(out, dpi=150)
print("wrote", out)
