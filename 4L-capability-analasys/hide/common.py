"""Shared helpers for the capability analysis: load results, vocab string
table + classification, context decoding."""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))  # project root, for load.py

EOS_ID = 0  # <|endoftext|>


@lru_cache(maxsize=1)
def results():
    return np.load(HERE / "results" / "predictions.npz")


@lru_cache(maxsize=1)
def tokenizer():
    import load
    return load.load_tokenizer("pile_4l")


VOCAB = 50277  # model vocab; tokenizer.vocab_size (50254) omits added specials


@lru_cache(maxsize=1)
def vocab_strings() -> list[str]:
    """Decoded string of every vocab id (50277 entries)."""
    return tokenizer().batch_decode([[i] for i in range(VOCAB)])


def classify(s: str, token_id: int | None = None) -> str:
    """String-level class of a token: eos / bytes / space / alpha_space /
    alpha_nospace / number / punct / mixed.
    (alpha_nospace may be a word-continuation or a line-initial word --
    distinguishing those needs the occurrence context, see occurrence_class.)"""
    if token_id == EOS_ID:
        return "eos"
    if "�" in s:
        return "bytes"
    if s.isspace() or s == "":
        return "space"
    core = s.strip()
    if core.isalpha():
        return "alpha_space" if s[0] == " " else "alpha_nospace"
    if any(c.isdigit() for c in core):
        return "number"
    if not any(c.isalnum() for c in core):
        return "punct"
    return "mixed"


@lru_cache(maxsize=1)
def vocab_classes() -> list[str]:
    return [classify(s, i) for i, s in enumerate(vocab_strings())]


def occurrence_class(token_id: int, prev_string: str) -> str:
    """Class of a token occurrence: splits alpha_nospace into word_cont
    (glued to a preceding alphanumeric) vs word_start (after whitespace etc.)."""
    c = vocab_classes()[token_id]
    if c == "alpha_nospace":
        glued = bool(prev_string) and (prev_string[-1].isalnum())
        return "word_cont" if glued else "word_start_nospace"
    return c


def decode_context(row: np.ndarray, t: int, n_tokens: int = 60) -> str:
    """Decoded text of the up-to-n_tokens context tokens preceding target t
    (i.e. inputs[max(0, t+1-n):t+1] -- the model predicts target t from
    inputs[:t+1])."""
    ids = row[max(0, t + 1 - n_tokens) : t + 1]
    return tokenizer().decode(ids.tolist())
