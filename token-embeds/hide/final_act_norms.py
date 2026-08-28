"""Histograms of final-residual-stream activation norms, both models, over
the harvested training-data positions (final_acts.py). Analog of plot.py for
embedding norms. Top row: after ln_f (just before the unembedding); bottom
row: the same positions just BEFORE ln_f.

(For scale: RMSNorm rescales every vector to ||x_hat|| = sqrt(d), so post-norm
||h|| = ||g * x_hat|| and sqrt(d) — printed, not drawn — is the norm if all
ln_f gains g were 1. Pre-norm the scale is unconstrained.) Next to each
histogram: the 10 highest- / lowest-norm positions, shown as [token] with a
few tokens of preceding context.

Output: token-embeds/final_act_norms.png
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

OUT = HERE.parent / "final_act_norms.png"

BAR = "#4e79a7"

pile_tok = load_tokenizer("pile_4l")
simple_tok = load_tokenizer("simple_2l")

panels = [
    ("pile_4l  (d=768, 102,400 positions)", "final_acts_pile_4l.npz", 768, pile_tok),
    ("simple_2l  (d=192, 99,615 positions)", "final_acts_simple_2l.npz", 192, simple_tok),
]


def extremes_text(norms, token_ids, row, pos, tok, k: int = 10, ctx: int = 3) -> str:
    """Top/bottom-k norm positions as  norm  'context'['token']."""

    def show(i: int) -> str:
        r, p = row[i], pos[i]
        prev = token_ids[(row == r) & (pos >= p - ctx) & (pos < p)]
        context = tok.decode(prev.tolist()) if len(prev) else ""
        return f"{repr(context)}[{repr(tok.decode([int(token_ids[i])]))}]"

    order = np.argsort(-norms)
    lines = ["highest norm:"]
    lines += [f" {norms[i]:6.2f}  {show(i)}" for i in order[:k]]
    lines += ["", "lowest norm:"]
    lines += [f" {norms[i]:6.2f}  {show(i)}" for i in order[::-1][:k]]
    return "\n".join(lines)


fig, axes = plt.subplots(2, 4, figsize=(17.5, 9.2), width_ratios=[3, 2.4, 3, 2.4])
for col, (title, cache, d, tok) in enumerate(panels):
    data = np.load(HERE / "cache" / cache)
    token_ids, row, pos = data["token_ids"], data["row"], data["pos"]
    for r, (key, where) in enumerate([("acts", "after ln_f"), ("acts_pre", "before ln_f")]):
        ax, text_ax = axes[r, 2 * col], axes[r, 2 * col + 1]
        norms = np.linalg.norm(data[key], axis=1)

        ax.hist(norms, bins=100, color=BAR, edgecolor="none")
        ax.set_title(f"{title}\n{where}" if r == 0 else where, fontsize=11)
        ax.set_xlabel(f"‖activation {where}‖₂")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        text_ax.axis("off")
        text_ax.text(
            0.0, 1.0, extremes_text(norms, token_ids, row, pos, tok),
            transform=text_ax.transAxes, ha="left", va="top",
            fontsize=6.5, family="monospace", color="#333333",
        )
        print(f"{title} {where}:  median {np.median(norms):.3f}, mean {norms.mean():.3f}, "
              f"min {norms.min():.3f}, max {norms.max():.3f}, sqrt(d) {math.sqrt(d):.3f}")
axes[0, 0].set_ylabel("token positions")
axes[1, 0].set_ylabel("token positions")

fig.suptitle("Final-residual-stream activation norms on training data", y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
