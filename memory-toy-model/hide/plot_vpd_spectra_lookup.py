"""MLP-CI vs lookup-table-CI mean-CI spectra, k15 cells -> vpd_mean_ci_lookup.png.

1x3 panels (arch); solid = the chosen MLP-CI decomposition, dashed = the lookup-CI
diagnostic of the same target at the same minimality coefficient. Same axes convention
as vpd_mean_ci.png. Local, ~2 s.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
mlp = np.load(HERE / "cache" / "vpd_mean_ci.npz")
lookup = np.load(HERE / "cache" / "vpd_mean_ci_lookup.npz")

SITE_COLORS = {
    "q_proj": "#2a78d6", "k_proj": "#7db3e8", "v_proj": "#1baf7a",
    "o_proj": "#84d6b8", "mix": "#8a5cd6", "c_fc": "#d6552a", "down_proj": "#e8a37d",
}
K = 15

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, arch in zip(axes, ["attn", "twoemb", "mix"]):
    sites = sorted({key.split("|")[2] for key in lookup.files
                    if key.startswith(f"{arch}|{K}|")})
    for site in sites:
        for data, style, tag in ((mlp, "-", "MLP CI"), (lookup, "--", "lookup CI")):
            values = np.sort(data[f"{arch}|{K}|{site}"])[::-1]
            n_alive = int((values > 1e-6).sum())
            ax.plot(np.arange(1, len(values) + 1), np.maximum(values, 1e-12),
                    style, color=SITE_COLORS[site], lw=1.4,
                    label=f"{site} {tag} (alive {n_alive})")
    ax.axhline(1e-6, color="green", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 2)
    ax.set_title(f"{arch}, n = 2^{K}", fontsize=11)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_xlabel("component rank (by mean CI, desc)")
axes[0].set_ylabel("mean CI (lower_leaky, full training set)")
fig.suptitle("k15 decompositions: MLP CI fn (solid, C=192/768) vs trained lookup-table CI "
             "(dashed, C=1600 all matrices); all matrices rank 96", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE.parent / "vpd_mean_ci_lookup.png"
fig.savefig(out, dpi=150)
print("wrote", out)
