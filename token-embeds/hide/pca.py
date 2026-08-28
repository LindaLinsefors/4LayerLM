"""PCA of the alive-token embeddings, per model.

Top panel: variance along each PC (eigenvalue spectrum), both models.
Bottom panels (one per model): mean projection of the RAW (non-centered)
embeddings onto each alive/frequent-PCA direction, for three token groups.
pile_4l (categories per CLAUDE.md 2026-08-27): frequent (count >= 7),
unfrequent (rest, excluding missing), missing (the 304 never-in-training
tokens from missing_tokens.npy). simple_2l: alive (freq >= 10), dead-but-seen,
never seen. Each PC's arbitrary sign is fixed so the alive/frequent group's
mean is >= 0; relative signs between groups on the same PC are preserved.

Alive tokens (CLAUDE.md, 2026-08-27): pile_4l count >= 7 over the 4,000 cached
Pile rows; simple_2l freq >= 10 in the token table.

Output: token-embeds/pca.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent.parent / "pca.png"

pile_emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
simple_emb = torch.load(
    ROOT / "prev_paper/models/simplestories_2layer/target_model_gf6rbga0/model_step_99999.pt",
    map_location="cpu", weights_only=True,
)["wte.weight"].float().numpy()

pile_rows = torch.stack(torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
))
pile_counts = np.bincount(pile_rows.flatten().numpy(), minlength=len(pile_emb))

table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
simple_counts = np.zeros(len(simple_emb), dtype=np.int64)
simple_counts[table["id"].to_numpy()] = table["freq"].to_numpy()

MODELS = [
    ("pile_4l", pile_emb, pile_counts, 7),
    ("simple_2l", simple_emb, simple_counts, 10),
]
MODEL_COLORS = {"pile_4l": "#4e79a7", "simple_2l": "#e0793d"}
GROUP_COLORS = {"alive": "#4e79a7", "dead, seen": "#e0793d", "never seen": "#a1443a",
                "frequent": "#4e79a7", "unfrequent": "#e0793d", "missing": "#a1443a"}

missing_ids = np.load(Path(__file__).parent / "missing_tokens.npy")

fig, ((ax_spec, ax_spec2), (ax_p, ax_s)) = plt.subplots(2, 2, figsize=(12, 8.5))
ax_spec2.remove()
ax_spec.set_position([0.08, 0.56, 0.87, 0.38])  # spectrum spans the full top row
mean_axes = {"pile_4l": ax_p, "simple_2l": ax_s}

for name, emb, counts, thr in MODELS:
    alive = counts >= thr
    x = emb[alive] - emb[alive].mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s**2 / alive.sum()

    ax_spec.plot(np.arange(1, len(var) + 1), var,
                 color=MODEL_COLORS[name], linewidth=2)
    ax_spec.annotate(
        f"{name}  ({alive.sum():,} "
        f"{'frequent' if name == 'pile_4l' else 'alive'} tokens, d={emb.shape[1]})",
        xy=(len(var), var[-1]),
        xytext=(6, 0) if name == "simple_2l" else (-4, 8),
        textcoords="offset points",
        ha="left" if name == "simple_2l" else "right",
        va="center" if name == "simple_2l" else "bottom",
        fontsize=9, color=MODEL_COLORS[name],
    )
    top = ", ".join(f"{v:.3f}" for v in var[:3])
    print(f"{name}: total var {var.sum():.3f}, top PCs [{top}], "
          f"PCs for 50%/90% of var: {np.searchsorted(np.cumsum(var) / var.sum(), [0.5, 0.9]) + 1}")

    # mean projection per group onto each PC: raw embeddings, no centering
    proj = emb @ vt.T
    alive_mean = proj[alive].mean(axis=0)
    proj *= np.where(alive_mean >= 0, 1.0, -1.0)  # sign: alive mean >= 0
    ax = mean_axes[name]
    pcs = np.arange(1, len(var) + 1)
    if name == "pile_4l":
        missing = np.zeros(len(emb), dtype=bool)
        missing[missing_ids] = True
        groups = [("frequent", alive), ("unfrequent", ~alive & ~missing),
                  ("missing", missing)]
    else:
        groups = [("alive", alive), ("dead, seen", (counts > 0) & ~alive),
                  ("never seen", counts == 0)]
    for label, sel in groups:
        ax.plot(pcs, proj[sel].mean(axis=0), color=GROUP_COLORS[label],
                linewidth=1.2, label=f"{label}  (n={sel.sum():,})")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("principal component index")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

ax_spec.set_xlabel("principal component index")
ax_spec.set_ylabel("variance along PC (eigenvalue)")
ax_spec.set_yscale("log")
ax_spec.set_title("PCA spectrum of frequent/alive-token embeddings")
ax_spec.grid(alpha=0.25, linewidth=0.5)
ax_spec.spines[["top", "right"]].set_visible(False)
ax_p.set_ylabel("group-mean projection of raw embeddings onto PC\n(sign: frequent/alive mean ≥ 0)")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
