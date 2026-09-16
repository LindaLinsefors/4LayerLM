"""Shared helpers for the eos-isolation experiments.

Data unit: a mid-sequence <|endoftext|> "event" (row_index, p) in the 4,000
cached Pile rows — rows[row_index][p] == EOS_ID with

    P_MIN <= p <= P_MAX      (>= 64 tokens of context before the EOS)
    no second EOS in p+1..p+D  (offsets d = 1..D after the boundary all score
                                tokens of a single fresh document)

Replacement contexts come from "partner" events: the p tokens immediately
before another row's EOS are an equally long context that ends at a genuine
document end, so [X', EOS] stays in-distribution.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

EOS_ID = 0    # <|endoftext|> (GPT-NeoX tokenizer)
NL_ID = 187   # '\n'
T = 512       # model context (cached rows are 513 tokens; last one dropped)
D = 64        # offsets after the boundary that get scored
P_MIN = 64
P_MAX = T - 1 - D  # 447

# Fixed color assignment (Okabe-Ito, CVD-safe), same in every figure:
C_EOS = "#D55E00"   # vermillion — EOS-separator conditions / the EOS key
C_NL = "#0072B2"    # blue       — newline-separator conditions
C_CEIL = "#000000"  # black      — mid-document swap ceiling
C_NONE = "#777777"  # gray       — minimal-context reference
C_PRE = "#CC79A7"   # purple     — pre-boundary content keys
C_KEY0 = "#E69F00"  # orange     — position-0 key
C_DOC = "#009E73"   # green      — own-document keys


def load_rows() -> torch.Tensor:
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)
    return torch.stack([r[:T] for r in rows])


def find_events(rows: torch.Tensor) -> list[tuple[int, int]]:
    e = (rows == EOS_ID).numpy()
    events = []
    for r in range(len(e)):
        for p in np.nonzero(e[r])[0]:
            if P_MIN <= p <= P_MAX and not e[r, p + 1:p + 1 + D].any():
                events.append((r, int(p)))
    return events


def assign_partners(events, rng, need_p=None, own_rows=None) -> np.ndarray:
    """partner[i] = index j of an event with p_j >= need_p[i] and a row different
    from own_rows[i], so rows[r_j][p_j - need : p_j] is a document ending of the
    required length. Defaults: need_p = the events' own p, own_rows = their own
    rows. -1 where no partner exists."""
    ps = np.array([p for _, p in events])
    rows_of = np.array([r for r, _ in events])
    if need_p is None:
        need_p = ps
    if own_rows is None:
        own_rows = rows_of
    out = np.full(len(need_p), -1, dtype=int)
    for i, need in enumerate(need_p):
        cand = np.nonzero((ps >= need) & (rows_of != own_rows[i]))[0]
        if len(cand):
            out[i] = int(rng.choice(cand))
    return out


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def attn_probs(attn, x):
    """(B, H, T', T') attention probabilities, replicating the module forward
    (the model runs flash attention, so A is never materialized)."""
    B, T_, _ = x.shape
    q = attn.q_proj(x).view(B, T_, attn.n_head, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(B, T_, attn.n_key_value_heads, attn.head_dim).transpose(1, 2)
    pos = torch.arange(T_, device=x.device).unsqueeze(0)
    cos = attn.rotary_cos[pos].to(q.dtype)
    sin = attn.rotary_sin[pos].to(q.dtype)
    q, k = attn._apply_rotary_pos_emb(q, k, cos, sin)
    if attn.repeat_kv_heads > 1:
        k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
    att = (q @ k.transpose(-2, -1)) / np.sqrt(attn.head_dim)
    att = att.masked_fill(attn.bias[:, :, :T_, :T_] == 0, float("-inf"))
    return F.softmax(att, dim=-1)
