"""Shared data + styling for the simple-emb analysis.

Provides the SimpleStories token-embedding matrix (4019 x 192; wte, which is
tied to lm_head), token strings, a coarse token classification, and corpus
token frequencies (counted over a sample of training stories). Everything is
cached under simple-emb-clusters/cache/.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # project root, for load.py

from load import SIMPLE_2L, load_tokenizer, simple_samples  # noqa: E402

CACHE = HERE / "cache"
FIGURES = HERE / "figures"
CACHE.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------- data loading


def embeddings_and_tokens() -> tuple[np.ndarray, list[str]]:
    """(emb (4019, 192) float32, token strings). wte.weight; tied to lm_head."""
    emb_f, tok_f = CACHE / "emb.npy", CACHE / "tokens.json"
    if emb_f.exists() and tok_f.exists():
        return np.load(emb_f), json.loads(tok_f.read_text(encoding="utf-8"))

    import torch

    state = torch.load(
        SIMPLE_2L / "target_model_gf6rbga0" / "model_step_99999.pt",
        map_location="cpu", weights_only=True,
    )
    assert torch.equal(state["wte.weight"], state["lm_head.weight"])
    emb = state["wte.weight"].numpy().astype(np.float32)

    tok = load_tokenizer("simple_2l")
    tokens = tok.convert_ids_to_tokens(list(range(emb.shape[0])))

    np.save(emb_f, emb)
    tok_f.write_text(json.dumps(tokens), encoding="utf-8")
    return emb, tokens


# Token classes, in fixed display order (= palette slot order).
CLASS_NAMES = ["word", "##suffix", "Capitalized", "punctuation", "number", "special"]


def token_classes(tokens: list[str]) -> np.ndarray:
    """Coarse class index per token (into CLASS_NAMES)."""
    classes = np.empty(len(tokens), dtype=np.int64)
    for i, t in enumerate(tokens):
        if t in ("[UNK]", "[EOS]", "[PAD]", "[BOS]"):
            c = "special"
        elif t.startswith("##"):
            c = "##suffix"
        elif any(ch.isdigit() for ch in t):
            c = "number"
        elif not any(ch.isalpha() for ch in t):
            c = "punctuation"
        elif t[0].isupper():
            c = "Capitalized"
        else:
            c = "word"
        classes[i] = CLASS_NAMES.index(c)
    return classes


def frequencies(vocab_size: int, n_stories: int = 1500) -> np.ndarray:
    """Token counts over `n_stories` training stories (datasets-server sample).

    [EOS] is stripped by simple_samples, so its count here is 0 even though it
    appears once per story in training.
    """
    f = CACHE / "freq.npy"
    if f.exists():
        return np.load(f)

    import time

    counts = np.zeros(vocab_size, dtype=np.int64)
    for offset in range(0, n_stories, 100):
        for attempt in range(5):
            try:
                rows = simple_samples(100, offset)
                break
            except Exception as e:  # datasets-server throws transient 502s
                print(f"  offset {offset}: {e}; retrying", flush=True)
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"datasets-server kept failing at offset {offset}")
        for row in rows:
            np.add.at(counts, row.numpy(), 1)
        print(f"  counted stories up to offset {offset + 100}", flush=True)
    np.save(f, counts)
    return counts


# ------------------------------------------------------------------- styling
# Palette + chart chrome from the validated dataviz reference palette (light).

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# Categorical slots 1-6, in fixed order -> CLASS_NAMES order.
CLASS_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

# Sequential blue ramp, steps 100 -> 700 (light -> dark).
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blueseq", SEQ_RAMP)


def style_ax(ax, square=True):
    """Recessive chrome: light surface, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    if square:
        ax.set_aspect("equal")


def new_fig(*args, **kwargs):
    import matplotlib.pyplot as plt

    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig


def save_fig(fig, name: str):
    out = FIGURES / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out.relative_to(HERE.parent)}")
