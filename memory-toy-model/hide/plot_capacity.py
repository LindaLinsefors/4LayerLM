"""Capacity curves for the three memory-toy architectures.

Reads hide/cache/sweep_summary[_<arch>].json -> memory-toy-model/capacity.png:
left panel = fraction of facts memorized vs dataset size, right panel = total
stored bits acc * n * log2(1024) vs dataset size, with 2 bits/param lines per
arch. Local, ~1 s.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V, D, M = 1024, 96, 384

# parameter counts (matrices + gains + sinks)
P_ATTN = 2 * V * D + 4 * D * D + 2 * D * M + 3 * D + 3
P_TWOEMB = 3 * V * D + 2 * D * M + 2 * D
P_MIX = 2 * V * D + D * D + 2 * D * M + 2 * D
ARCHS = [("attn", "sweep_summary.json", P_ATTN, "#2a78d6"),
         ("twoemb", "sweep_summary_twoemb.json", P_TWOEMB, "#1baf7a"),
         ("mix", "sweep_summary_mix.json", P_MIX, "#d6552a")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for arch, fname, n_params, color in ARCHS:
    rows = json.loads((HERE / "cache" / fname).read_text())
    ks = np.array([r["k"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    n = 2.0 ** ks
    ax1.plot(ks, acc, "o-", color=color, label=f"{arch} ({n_params/1e3:.0f}k params)")
    ax2.plot(ks, acc * n * 10, "o-", color=color, label=arch)
    ax2.axhline(2 * n_params, color=color, ls=":", lw=1)

ax1.set_xlabel("log2(n facts)")
ax1.set_ylabel("fraction memorized (top-1)")
ax1.set_ylim(0, 1.02)
ax1.legend()
ax1.grid(alpha=0.3)
ax2.plot(ks, 10 * n, color="gray", ls="--", lw=1, label="all facts (10 bits each)")
ax2.set_xlabel("log2(n facts)")
ax2.set_ylabel("stored bits = acc · n · 10")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title("dotted lines: 2 bits/param per arch")
ax1.set_title("memorization vs dataset size")
fig.tight_layout()
out = HERE.parent / "capacity.png"
fig.savefig(out, dpi=150)
print("wrote", out)
