"""Mean-CI spectra of the memory-toy VPD decompositions -> vpd_mean_ci.png.

3x3 grid (rows = arch, cols = n facts). Per panel one line per decomposed matrix:
components sorted by descending mean CI (over the model's FULL training set) on the
x-axis, mean CI on the y-axis (log). Dashed line = the project's 1e-6 alive cutoff.
Reads hide/cache/vpd_mean_ci.npz (from vpd_spectra_modal.py). Local, ~2 s.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = np.load(HERE / "cache" / "vpd_mean_ci.npz")

ARCHS = ["attn", "twoemb", "mix"]
KS = [15, 16, 17]
SITE_COLORS = {
    "q_proj": "#2a78d6", "k_proj": "#7db3e8", "v_proj": "#1baf7a",
    "o_proj": "#84d6b8", "mix": "#8a5cd6", "c_fc": "#d6552a", "down_proj": "#e8a37d",
}
FLAGGED = {("attn", 17), ("mix", 17)}  # poor-recon cells (see vpd_report.md)

fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
for i, arch in enumerate(ARCHS):
    for j, k in enumerate(KS):
        ax = axes[i, j]
        sites = sorted({key.split("|")[2] for key in data.files
                        if key.startswith(f"{arch}|{k}|")})
        for site in sites:
            values = np.sort(data[f"{arch}|{k}|{site}"])[::-1]
            n_alive = int((values > 1e-6).sum())
            rank = 96  # every site reads or writes the 96-dim stream: rank = min(d_in, d_out) = 96
            ax.plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-12),
                    color=SITE_COLORS[site], lw=1.4,
                    label=f"{site} (rank {rank}, C={len(values)}, alive {n_alive})")
        ax.axhline(1e-6, color="green", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 2)
        flag = "  ⚠ poor recon" if (arch, k) in FLAGGED else ""
        ax.set_title(f"{arch}, n = 2^{k}{flag}", fontsize=11)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25)
        if i == 2:
            ax.set_xlabel("component rank (by mean CI, desc)")
        if j == 0:
            ax.set_ylabel("mean CI (lower_leaky, full training set)")
fig.suptitle("Memory-toy VPD: per-component mean CI spectra "
             "(each model's full fact table; dashed = 1e-6 alive cutoff)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE.parent / "vpd_mean_ci.png"
fig.savefig(out, dpi=150)
print("wrote", out)
