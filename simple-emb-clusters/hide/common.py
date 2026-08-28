"""Shared data + styling for the simple-emb analysis (alive tokens, de-frequencied).

The object of study is the SimpleStories token-embedding matrix (wte, tied to
lm_head), restricted to the ALIVE tokens (corpus freq >= 10 in
simple-token-table/token_table.csv; n = 3702 of 4019) and with the frequency
direction removed. Two removal variants, both applied to the mean-centered
alive rows A~ = A - mean(A):

  pc1  -- project out the top principal component of A~
  vcov -- project out v ∝ A~^T y~, the covariance direction with
          y = log(count) (zeroes the covariance of every remaining direction
          with log count; see token-embeds/freq_direction.py + CLAUDE.md)

cos(pc1, vcov) ≈ 0.98, so the variants are expected to be near-identical;
run_compare.py quantifies the difference. Per-variant outputs live in
cache/<variant>/ and figures/<variant>/. Raw-embedding caches (emb.npy,
tokens.json) are shared in cache/.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))  # project root, for load.py

from load import SIMPLE_2L, load_tokenizer  # noqa: E402

CACHE = HERE / "cache"
FIGURES = HERE.parent / "figures"
CACHE.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

VARIANTS = ("pc1", "vcov")
ALIVE_THR = 10  # CLAUDE.md 2026-08-27: simple_2l alive = freq >= 10


def cache_dir(variant: str) -> Path:
    d = CACHE / variant
    d.mkdir(exist_ok=True)
    return d


def fig_dir(variant: str) -> Path:
    d = FIGURES / variant
    d.mkdir(exist_ok=True)
    return d


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


def frequencies(vocab_size: int) -> np.ndarray:
    """Corpus token counts from simple-token-table/token_table.csv (20k stories,
    5.74M tokens). [EOS] reads 0 (stories were encoded without it)."""
    import pandas as pd

    table = pd.read_csv(HERE.parent.parent / "simple-token-table" / "token_table.csv")
    counts = np.zeros(vocab_size, dtype=np.int64)
    counts[table["id"].to_numpy()] = table["freq"].to_numpy()
    return counts


def processed(variant: str) -> SimpleNamespace:
    """Alive-token embeddings with the frequency direction removed.

    Returns namespace with:
      X       (n_alive, 192) centered rows, freq direction projected out
      Xn      row-normalized X (for cosine / clustering)
      ids     original token ids of the alive tokens
      tokens  their strings
      classes token-class index per alive token (into CLASS_NAMES)
      freq    their corpus counts
      v       the removed unit direction; mu = the subtracted mean
    """
    assert variant in VARIANTS, variant
    emb, all_tokens = embeddings_and_tokens()
    all_freq = frequencies(len(all_tokens))
    alive = all_freq >= ALIVE_THR
    ids = np.where(alive)[0]

    A = emb[alive].astype(np.float64)
    mu = A.mean(0)
    Ac = A - mu
    y = np.log(all_freq[alive].astype(np.float64))
    yc = y - y.mean()

    if variant == "pc1":
        _, _, vt = np.linalg.svd(Ac, full_matrices=False)
        v = vt[0]
    else:
        v = Ac.T @ yc
    v /= np.linalg.norm(v)
    if (Ac @ v) @ yc < 0:  # sign: projection correlates positively with freq
        v = -v

    X = (Ac - np.outer(Ac @ v, v)).astype(np.float32)
    tokens = [all_tokens[i] for i in ids]
    return SimpleNamespace(
        X=X, Xn=X / np.linalg.norm(X, axis=1, keepdims=True),
        ids=ids, tokens=tokens, classes=token_classes(tokens),
        freq=all_freq[ids], v=v.astype(np.float32), mu=mu.astype(np.float32),
    )


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


def save_fig(fig, name: str, variant: str | None = None):
    out = (fig_dir(variant) if variant else FIGURES) / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out.relative_to(HERE.parent)}")
