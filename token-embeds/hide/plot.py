"""Histograms of token-embedding row norms for both models, with the mean
embedding norm at initialization marked as a vertical line, and the 10
highest- / lowest-norm tokens listed next to each histogram.

Init scheme (param-decomp pretrain, llama_simple_mlp.py): wte ~ N(0, 0.02^2)
i.i.d. (lm_head has LLMC_SKIP_INIT and is tied to wte, so the Embedding init
is what sticks). The norm of a row is then 0.02 * chi(d), with mean

    E||e|| = 0.02 * sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)  ~=  0.02 * sqrt(d).

Output: token-embeds/embedding_norms.png
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

OUT = Path(__file__).parent.parent / "embedding_norms.png"

INIT_STD = 0.02


def chi_mean(d: int) -> float:
    return math.sqrt(2) * math.exp(math.lgamma((d + 1) / 2) - math.lgamma(d / 2))


def wte_norms(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return state["wte.weight"].float().norm(dim=1)


pile_state = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)
simple_state = torch.load(
    ROOT / "prev_paper/models/simplestories_2layer/target_model_gf6rbga0/model_step_99999.pt",
    map_location="cpu",
    weights_only=True,
)

pile_tok = load_tokenizer("pile_4l")
simple_tok = load_tokenizer("simple_2l")

# Token display: decode() for the pile BPE (repr shows the leading-space
# convention); convert_ids_to_tokens() for SimpleStories WordPiece (keeps ##).
panels = [
    ("pile_4l  (d=768, vocab 50,277)", wte_norms(pile_state), 768,
     lambda i: repr(pile_tok.decode([i]))),
    ("simple_2l  (d=192, vocab 4,019)", wte_norms(simple_state), 192,
     lambda i: repr(simple_tok.convert_ids_to_tokens(i))),
]

BAR = "#4e79a7"
REF = "#b3502d"


def extremes_text(norms: torch.Tensor, show, k: int = 10) -> str:
    order = norms.argsort(descending=True)
    lines = ["highest norm:"]
    lines += [f" {norms[i]:.3f}  {show(int(i))}" for i in order[:k]]
    lines += ["", "lowest norm:"]
    lines += [f" {norms[i]:.3f}  {show(int(i))}" for i in reversed(order[-k:])]
    return "\n".join(lines)


fig, axes = plt.subplots(
    1, 5, figsize=(17.5, 4.6), width_ratios=[3, 3, 1.5, 3, 1.5]
)
pile_zoom_ax = axes[1]
hist_text_pairs = [(axes[0], axes[2]), (axes[3], axes[4])]
for (ax, text_ax), (title, norms, d, show) in zip(hist_text_pairs, panels):
    init_mean = INIT_STD * chi_mean(d)
    ax.hist(norms.numpy(), bins=100, color=BAR, edgecolor="none")
    ax.axvline(init_mean, color=REF, linestyle="--", linewidth=1.5)
    ax.annotate(
        f"mean norm at init\n0.02·E[χ({d})] = {init_mean:.3f}",
        xy=(init_mean, 0.97), xycoords=("data", "axes fraction"),
        xytext=(6, 0), textcoords="offset points",
        ha="left", va="top", fontsize=9, color=REF,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("‖embedding row‖₂")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    text_ax.axis("off")
    text_ax.text(
        0.0, 1.0, extremes_text(norms, show),
        transform=text_ax.transAxes, ha="left", va="top",
        fontsize=7.5, family="monospace", color="#333333",
    )
    print(f"{title}:  median {norms.median():.3f}, mean {norms.mean():.3f}, "
          f"min {norms.min():.3f}, max {norms.max():.3f}, init mean {init_mean:.4f}")
axes[0].set_ylabel("tokens")

# Zoomed copy of the pile histogram: y-axis cropped to 100 to show the tails.
pile_norms = panels[0][1]
pile_zoom_ax.hist(pile_norms.numpy(), bins=100, color=BAR, edgecolor="none")
pile_zoom_ax.axvline(INIT_STD * chi_mean(768), color=REF, linestyle="--", linewidth=1.5)
pile_zoom_ax.set_ylim(0, 100)
pile_zoom_ax.set_title("pile_4l  (zoom, y ≤ 100)", fontsize=11)
pile_zoom_ax.set_xlabel("‖embedding row‖₂")
pile_zoom_ax.grid(axis="y", alpha=0.25, linewidth=0.5)
pile_zoom_ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Token embedding norms after training vs. initialization", y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
