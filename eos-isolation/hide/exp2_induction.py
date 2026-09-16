"""Experiment 2 — cross-EOS induction: does information leak through the
boundary when the pre-EOS context is maximally useful?

Natural pre-EOS context is useless for the next document (the Pile is
shuffled), so Experiment 1 can't distinguish "the model firewalls the
boundary" from "there was nothing worth reading". Here the incentive is
maximal: the pre-boundary context IS the document being predicted.

For N real 64-token document openings D (tokens p+1..p+64 after a cached-row
EOS) and unrelated openings D' (another event's document), score the second
copy of D in five inputs:

    same + EOS :  [D,  EOS,  D]     other + EOS :  [D', EOS,  D]
    same + \n  :  [D,  '\n', D]     other + \n  :  [D', '\n', D]
    none       :  [EOS, D]          (minimal-context reference, 65 tokens)

At offset d = 1..64 (position 64+d; logits at 64+d-1) record NLL(D_d) and
top-1 accuracy. With mean NLLs L(prefix, sep) over d = 2..64 (d = 1 has no
induction cue), the leak fraction

    leak = [L(other,EOS) - L(same,EOS)] / [L(other,\n) - L(same,\n)]

is 0 for a perfect firewall and 1 for no firewall at all. Top-1 accuracy in
the `same` conditions is the verbatim copy rate.

Cache: hide/cache/exp2.npz (delete to recompute). Figure: ../exp2_induction.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (C_EOS, C_NL, C_NONE, D, EOS_ID, HERE, NL_ID, ROOT,
                    find_events, load_rows, style)

import sys
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 16
CACHE = HERE / "cache" / "exp2.npz"
CONDS = ["same_eos", "same_nl", "other_eos", "other_nl", "none"]


def compute():
    rows = load_rows()
    events = find_events(rows)
    n = len(events)
    docs = torch.stack([rows[r, p + 1:p + 1 + D] for r, p in events])  # (n, 64)

    # unrelated opening for each doc: another event's doc from a different row
    other = np.arange(n)
    for i in range(n):
        j = (i + 37) % n
        while events[j][0] == events[i][0]:
            j = (j + 1) % n
        other[i] = j
    print(f"{n} document openings")

    sep_e = torch.full((n, 1), EOS_ID, dtype=torch.long)
    sep_n = torch.full((n, 1), NL_ID, dtype=torch.long)
    seqs = {
        "same_eos": torch.cat([docs, sep_e, docs], 1),
        "same_nl": torch.cat([docs, sep_n, docs], 1),
        "other_eos": torch.cat([docs[other], sep_e, docs], 1),
        "other_nl": torch.cat([docs[other], sep_n, docs], 1),
        "none": torch.cat([sep_e, docs], 1),
    }

    model, _, _ = load_pile_4l()
    model = model.to(DEVICE)

    nll = np.zeros((len(CONDS), n, D), dtype=np.float32)
    acc = np.zeros((len(CONDS), n, D), dtype=bool)
    with torch.no_grad():
        for ci, cond in enumerate(CONDS):
            s = seqs[cond]
            start = 0 if cond == "none" else D  # first scored logits index
            for b0 in range(0, n, BS):
                batch = s[b0:b0 + BS].to(DEVICE)
                lp = torch.log_softmax(
                    model(batch)[:, start:start + D].float(), -1)
                true = docs[b0:b0 + BS].to(DEVICE)
                nll[ci, b0:b0 + BS] = (-lp.gather(-1, true[:, :, None])[:, :, 0]
                                       ).cpu().numpy()
                acc[ci, b0:b0 + BS] = (lp.argmax(-1) == true).cpu().numpy()
            print(f"  {cond}: mean NLL {nll[ci].mean():.3f}")

    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, nll=nll, acc=acc,
                        ps=np.array([p for _, p in events]))
    print(f"saved {CACHE}")


if not CACHE.exists():
    compute()
z = np.load(CACHE)
nll, acc = z["nll"], z["acc"]
n = nll.shape[1]
ds = np.arange(1, D + 1)
sl = slice(1, D)  # d = 2..64 (d=1 has no induction cue)

L = {c: nll[i, :, sl].mean() for i, c in enumerate(CONDS)}
per_doc = {c: nll[i, :, sl].mean(1) for i, c in enumerate(CONDS)}
gap_eos = L["other_eos"] - L["same_eos"]
gap_nl = L["other_nl"] - L["same_nl"]
leak = gap_eos / gap_nl
rng = np.random.default_rng(0)
boots = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    ge = per_doc["other_eos"][idx].mean() - per_doc["same_eos"][idx].mean()
    gn = per_doc["other_nl"][idx].mean() - per_doc["same_nl"][idx].mean()
    boots.append(ge / gn)
lo, hi = np.percentile(boots, [2.5, 97.5])

print(f"\nmean NLL on the second copy (d = 2..{D}), {n} docs:")
print(f"                 prefix = same doc   prefix = other doc   gap (other-same)")
print(f"  sep = EOS        {L['same_eos']:.4f}            {L['other_eos']:.4f}"
      f"            {gap_eos:.4f}")
print(f"  sep = '\\n'       {L['same_nl']:.4f}            {L['other_nl']:.4f}"
      f"            {gap_nl:.4f}")
print(f"  none ([EOS,doc] at pos 0):  {L['none']:.4f}")
print(f"\nleak fraction = {leak:.4f}  (95% bootstrap CI [{lo:.4f}, {hi:.4f}])")
print(f"copy rate (top-1 acc, d=2..{D}): same+EOS {acc[0, :, sl].mean():.3f}, "
      f"same+'\\n' {acc[1, :, sl].mean():.3f}, other+EOS {acc[2, :, sl].mean():.3f}, "
      f"other+'\\n' {acc[3, :, sl].mean():.3f}, none {acc[4, :, sl].mean():.3f}")

# ---- figure ----
fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12.5, 4.6))
spec = [("same_eos", C_EOS, "-", "D + EOS + D (repeat across EOS)"),
        ("other_eos", C_EOS, "--", "D' + EOS + D (fresh across EOS)"),
        ("same_nl", C_NL, "-", "D + '\\n' + D (repeat, no EOS)"),
        ("other_nl", C_NL, "--", "D' + '\\n' + D (fresh, no EOS)"),
        ("none", C_NONE, ":", "[EOS, D] at position 0")]
for cond, c, ls, label in spec:
    i = CONDS.index(cond)
    ax_l.plot(ds, nll[i].mean(0), color=c, ls=ls, lw=1.8, label=label)
    ax_a.plot(ds, acc[i].mean(0), color=c, ls=ls, lw=1.8, label=label)
ax_l.set_xlabel("offset d into the second copy (tokens)")
ax_l.set_ylabel("mean NLL of true token  [nats]")
ax_l.set_title("loss on a document whose exact copy sits before the boundary", fontsize=11)
ax_l.legend(fontsize=8, frameon=False)
ax_a.set_xlabel("offset d into the second copy (tokens)")
ax_a.set_ylabel("top-1 accuracy (= copy rate for 'same')")
ax_a.set_title("verbatim copying across the boundary", fontsize=11)
ax_a.legend(fontsize=8, frameon=False)
for ax in (ax_l, ax_a):
    style(ax)
fig.suptitle(f"pile_4l — cross-EOS induction test ({n} document openings): "
             f"leak fraction {leak:.2f}", y=1.0)
fig.tight_layout()
fig.savefig(HERE.parent / "exp2_induction.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'exp2_induction.png'}")
