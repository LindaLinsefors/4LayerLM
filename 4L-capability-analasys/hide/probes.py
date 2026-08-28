"""Targeted capability probes: hand-built prompts testing specific skills.

Each prompt is run as <|endoftext|> + prompt (document start, as in training);
we look at the model's distribution for the NEXT token.

Suites:
  induction   -- copy accuracy on repeated random token sequences (quantitative)
  brackets    -- closing ), ", ], environments
  facts       -- world-knowledge completions
  agreement   -- grammatical minimal pairs, P(correct form) vs P(wrong form)
  sequences   -- counting, weekdays, months, alphabet
  arithmetic  -- small sums/products, digit and word form
  idioms      -- collocations and fixed phrases
  code        -- code/markup conventions
  structure   -- key: value / Q: A: format induction

Prints per-probe results and saves results/probes.json.
Runtime: seconds (single prompts, local GPU or CPU).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import load
from common import EOS_ID, HERE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, _, tok = load.load_pile_4l()
model = model.to(DEVICE)
J = {}


@torch.no_grad()
def next_dist(prompt: str) -> torch.Tensor:
    ids = [EOS_ID] + tok.encode(prompt)
    logits = model(torch.tensor([ids], device=DEVICE))
    return logits[0, -1].softmax(-1).cpu()


@torch.no_grad()
def cand_p(prompt: str, cand: str) -> float:
    """Joint probability of the (possibly multi-token) continuation `cand`."""
    ids = [EOS_ID] + tok.encode(prompt)
    cids = tok.encode(cand)
    logits = model(torch.tensor([ids + cids[:-1]], device=DEVICE))[0]
    logp = logits[len(ids) - 1 :].log_softmax(-1)
    return float(logp[torch.arange(len(cids)), torch.tensor(cids, device=DEVICE)].sum().exp())


def top5(p: torch.Tensor) -> list:
    v, i = p.topk(5)
    return [(tok.decode([j]), round(float(x), 3)) for j, x in zip(i.tolist(), v)]


def probe(suite: str, prompt: str, expected: str):
    """P of `expected` as the continuation; rank if it is a single token."""
    p = next_dist(prompt)
    eids = tok.encode(expected)
    if len(eids) == 1:
        pe, rank = float(p[eids[0]]), int((p > p[eids[0]]).sum()) + 1
    else:
        pe, rank = cand_p(prompt, expected), None
    rec = {"prompt": prompt, "expected": expected, "p": round(pe, 4),
           "rank": rank, "top5": top5(p)}
    J.setdefault(suite, []).append(rec)
    print(f"  {prompt!r} -> {expected!r}: p={pe:.3f} rank={rank}  top: {rec['top5'][:3]}")


def pair(suite: str, prompt: str, good: str, bad: str):
    """Minimal pair: which of two continuations gets more probability."""
    pg, pb = cand_p(prompt, good), cand_p(prompt, bad)
    rec = {"prompt": prompt, "good": good, "bad": bad, "p_good": round(pg, 4),
           "p_bad": round(pb, 4), "correct": pg > pb,
           "ratio": round(pg / max(pb, 1e-9), 1)}
    J.setdefault(suite, []).append(rec)
    mark = "OK " if pg > pb else "XX "
    print(f"  {mark}{prompt!r}: {good!r} {pg:.3f} vs {bad!r} {pb:.3f}  (x{rec['ratio']})")


@torch.no_grad()
def induction_suite(n_seqs=20, seq_len=25, seed=0):
    """Repeat a random token sequence twice; measure top-1 accuracy on the
    second copy (pure induction: the only signal is the first copy)."""
    print("\n== induction (random repeated sequences) ==")
    rng = np.random.default_rng(seed)
    # candidate tokens: common " word"-style tokens for a natural-ish stream
    cand = [i for i in range(1000, 20000)
            if (s := tok.decode([i])).startswith(" ") and s[1:].isalpha()]
    acc1 = acc2 = n1 = n2 = 0
    for _ in range(n_seqs):
        seq = rng.choice(cand, seq_len, replace=False).tolist()
        ids = [EOS_ID] + seq + seq
        logits = model(torch.tensor([ids], device=DEVICE))[0]
        predicted = logits.argmax(-1).cpu()
        target = torch.tensor(ids[1:])
        hit = (predicted[:-1] == target)
        acc1 += hit[:seq_len].sum(); n1 += seq_len          # first copy: unpredictable
        acc2 += hit[seq_len + 1:].sum(); n2 += seq_len - 1  # second copy after 1st tok
        # (position seq_len predicts the start of the repeat -- also unpredictable)
    J["induction"] = {"first_copy_acc": float(acc1) / n1, "second_copy_acc": float(acc2) / n2,
                      "n_seqs": n_seqs, "seq_len": seq_len}
    print(f"  top-1 acc, first copy (no signal): {J['induction']['first_copy_acc']:.1%}")
    print(f"  top-1 acc, second copy (copyable): {J['induction']['second_copy_acc']:.1%}")


def main():
    induction_suite()

    print("\n== brackets & quotes ==")
    probe("brackets", 'He said, "I am tired', '"')
    probe("brackets", "The result (see Figure 3", ")")
    probe("brackets", "f(x", ")")
    probe("brackets", "The array indices [0, 1, 2", "]")
    probe("brackets", "\\begin{equation}\nE = mc^2\n\\end{", "equation")

    print("\n== facts ==")
    probe("facts", "The capital of France is", " Paris")
    probe("facts", "The Eiffel Tower is located in", " Paris")
    probe("facts", "Water is composed of hydrogen and", " oxygen")
    probe("facts", "The first President of the United States was George", " Washington")
    probe("facts", "The Earth revolves around the", " Sun")
    probe("facts", "Albert Einstein developed the theory of", " relativity")
    probe("facts", "The chemical symbol for gold is", " Au")
    probe("facts", "World War II ended in the year 19", "45")
    probe("facts", "The largest planet in the Solar System is", " Jupiter")
    probe("facts", "Romeo and", " Juliet")
    probe("facts", "The play Hamlet was written by William", " Shakespeare")
    probe("facts", "DNA stands for deoxyribonucleic", " acid")
    probe("facts", "President Barack", " Obama")

    print("\n== agreement (minimal pairs) ==")
    pair("agreement", "The key to the cabinets", " is", " are")
    pair("agreement", "The keys to the cabinet", " are", " is")
    pair("agreement", "The dog", " barks", " bark")
    pair("agreement", "The dogs", " bark", " barks")
    pair("agreement", "She", " has", " have")
    pair("agreement", "They", " have", " has")
    pair("agreement", "He", " was", " were")
    pair("agreement", "They", " were", " was")
    pair("agreement", "The girl hurt", " herself", " himself")
    pair("agreement", "The boys hurt", " themselves", " himself")
    pair("agreement", "Yesterday he", " walked", " walks")
    pair("agreement", "One of the", " books", " book")
    pair("agreement", "Each of the students", " was", " were")
    pair("agreement", "The scientists who made the discovery", " were", " was")

    print("\n== sequences ==")
    probe("sequences", "1, 2, 3, 4,", " 5")
    probe("sequences", "9, 10, 11,", " 12")
    probe("sequences", "one, two, three,", " four")
    probe("sequences", "Monday, Tuesday,", " Wednesday")
    probe("sequences", "January, February,", " March")
    probe("sequences", "a, b, c,", " d")
    probe("sequences", "2, 4, 6, 8,", " 10")

    print("\n== arithmetic ==")
    probe("arithmetic", "2 + 2 =", " 4")
    probe("arithmetic", "3 + 5 =", " 8")
    probe("arithmetic", "7 + 2 =", " 9")
    probe("arithmetic", "10 - 4 =", " 6")
    probe("arithmetic", "6 * 7 =", " 42")
    probe("arithmetic", "Two plus two equals", " four")
    probe("arithmetic", "2+2=", "4")

    print("\n== idioms & collocations ==")
    probe("idioms", "Once upon a", " time")
    probe("idioms", "as soon as", " possible")
    probe("idioms", "the United States of", " America")
    probe("idioms", "New York", " City")
    probe("idioms", "Ladies and", " gentlemen")
    probe("idioms", "in order", " to")
    probe("idioms", "at the end of the", " day")
    probe("idioms", "The quick brown fox jumps over the lazy", " dog")

    print("\n== code ==")
    probe("code", "import numpy as", " np")
    probe("code", "import matplotlib.pyplot as", " plt")
    probe("code", "for i in", " range")
    probe("code", "def main(", ")")
    probe("code", 'if __name__ == "__', "main")
    probe("code", "#include <std", "io")
    probe("code", '<a href="http', "://")
    probe("code", "SELECT * FROM", " table")

    print("\n== structure (format induction) ==")
    probe("structure", "Name: John\nAge: 30\nName: Mary\nAge:", " 2")
    probe("structure", "Q: What is your name?\nA: John.\nQ: Where do you live?\n", "A")
    probe("structure", "1. Introduction\n2. Methods\n3.", " Results")
    probe("structure", "- apples\n- oranges\n-", " bananas")

    with open(HERE / "results" / "probes.json", "w") as f:
        json.dump(J, f, indent=1)
    print("\nsaved results/probes.json")


if __name__ == "__main__":
    main()
