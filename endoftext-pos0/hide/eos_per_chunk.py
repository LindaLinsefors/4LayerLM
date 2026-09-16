"""Histogram of <|endoftext|> occurrences per training chunk (pile_4l).

Data: all 4,000 cached training rows (context-loss/hide/cache/pile_rows.pt) —
exact 513-token rows from danbraunai/pile-uncopyrighted-tok-shuffled, i.e. the
model's training chunks as-is (EOS appears only at document boundaries mid-row).

Output: endoftext-pos0/eos_per_chunk.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

EOS_ID = 0

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)
counts = np.array([(r == EOS_ID).sum().item() for r in rows])
n_rows, row_len = len(rows), len(rows[0])

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
bins = np.arange(counts.max() + 2) - 0.5
ax.hist(counts, bins=bins, color="tab:blue", edgecolor="white")
ax.set_xticks(np.arange(counts.max() + 1))
ax.set_xlabel("number of <|endoftext|> tokens in the chunk")
ax.set_ylabel(f"chunks (of {n_rows:,})")
ax.set_title(f"pile_4l training data — <|endoftext|> count per {row_len}-token chunk\n"
             f"mean {counts.mean():.2f}, zero-EOS fraction {np.mean(counts == 0):.1%}, max {counts.max()}")
for k in range(counts.max() + 1):
    n = int(np.sum(counts == k))
    if n:
        ax.annotate(f"{n}", (k, n), ha="center", va="bottom", fontsize=8)

out = HERE.parent / "eos_per_chunk.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
print(f"{n_rows} chunks x {row_len} tokens; EOS total {counts.sum()} "
      f"({counts.sum() / counts.size:.3f}/chunk); distribution: "
      + ", ".join(f"{k}: {np.sum(counts == k)}" for k in range(counts.max() + 1)))
