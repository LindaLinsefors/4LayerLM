"""How well does each model predict text it has already seen in its context?

Paradigm (per event): a natural-text span A (S tokens), a filler F_g of g
tokens from a different document, then A verbatim again:

    rep = [A, F_g, A]          ctl = [A', F_g, A]   (A' = unrelated span)

Scored on the second copy at offsets j = 0..S-1 (predicting A[j] from the
preceding context): per-offset NLL and top-1 accuracy. The control gives the
novel-text baseline at exactly matched positions, so

    copy benefit(j) = NLL_ctl(j) - NLL_rep(j)   [nats].

Models: pile_4l / sink45 / sink46 (S=64, gaps 0/64/192/384, spans from the
cached Pile rows, EOS-free) and simple_2l (S=32, gaps 0/32/96/192, spans from
the cached stories). Sink models via load_sink -> corrected RoPE.

Writes cache/copy_predict.npz. ~5 min local GPU. N=400 events per model.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import load

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 400
EOS_PILE = 0  # GPT-NeoX <|endoftext|>


def pile_material(rng):
    """(A, F, A2): EOS-free spans/fillers from distinct cached Pile rows."""
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
    rows = [r[:512] for r in rows]
    S, G = 64, 384

    def span(row_idx, length):
        r = rows[row_idx]
        for _ in range(50):
            s = rng.integers(0, len(r) - length + 1)
            seg = r[s:s + length]
            if (seg != EOS_PILE).all():
                return seg
        return None

    A, FF, A2 = [], [], []
    order = rng.permutation(len(rows))
    i = 0
    while len(A) < N:
        r0, r1, r2 = order[i % len(rows)], order[(i + 997) % len(rows)], order[(i + 1993) % len(rows)]
        i += 1
        a, f, a2 = span(r0, S), span(r1, G), span(r2, S)
        if a is None or f is None or a2 is None:
            continue
        A.append(a); FF.append(f); A2.append(a2)
    return torch.stack(A), torch.stack(FF), torch.stack(A2), S, [0, 64, 192, 384]


def story_material(rng):
    stories = torch.load(ROOT / "context-loss/hide/cache/simple_stories.pt")
    S, G = 32, 192
    long_enough = [s for s in stories if len(s) >= S]
    filler_pool = torch.cat([s for s in stories[:200]])

    A, FF, A2 = [], [], []
    order = rng.permutation(len(long_enough))
    i = 0
    while len(A) < N:
        r0, r2 = order[i % len(long_enough)], order[(i + 997) % len(long_enough)]
        i += 1
        s0, s2 = long_enough[r0], long_enough[r2]
        a = s0[(k := rng.integers(0, len(s0) - S + 1)):k + S]
        a2 = s2[(k := rng.integers(0, len(s2) - S + 1)):k + S]
        f0 = rng.integers(0, len(filler_pool) - G)
        A.append(a); FF.append(filler_pool[f0:f0 + G]); A2.append(a2)
    return torch.stack(A), torch.stack(FF), torch.stack(A2), S, [0, 32, 96, 192]


@torch.no_grad()
def score(model, seqs, p0, S):
    """Per-sequence NLL + top-1 hit at positions p0..p0+S-1 (targets seqs[:, p])."""
    nll = torch.empty(len(seqs), S)
    acc = torch.empty(len(seqs), S)
    for i in range(0, len(seqs), 32):
        b = seqs[i:i + 32].to(dev)
        logits = model(b)[:, p0 - 1:p0 + S - 1]           # predicts tokens at p0..p0+S-1
        tgt = b[:, p0:p0 + S]
        lp = F.log_softmax(logits.float(), dim=-1)
        nll[i:i + 32] = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
        acc[i:i + 32] = (logits.argmax(-1) == tgt).float().cpu()
    return nll, acc


out = {}
models = [
    ("pile_4l", lambda: load.load_pile_4l()[0], pile_material),
    ("sink45", lambda: load.load_sink(45)[0], pile_material),
    ("sink46", lambda: load.load_sink(46)[0], pile_material),
    ("simple_2l", lambda: load.load_simple_2l()[0], story_material),
]
for name, loader, material in models:
    rng = np.random.default_rng(0)  # same events for the three Pile models
    A, FF, A2, S, gaps = material(rng)
    model = loader().to(dev).eval()
    for g in gaps:
        rep = torch.cat([A, FF[:, :g], A], dim=1)
        ctl = torch.cat([A2, FF[:, :g], A], dim=1)
        p0 = S + g
        for cond, seqs in [("rep", rep), ("ctl", ctl)]:
            nll, acc = score(model, seqs, p0, S)
            out[f"{name}|{cond}|nll|{g}"] = nll.numpy()
            out[f"{name}|{cond}|acc|{g}"] = acc.numpy()
        print(f"{name} g={g:3d}: rep NLL {out[f'{name}|rep|nll|{g}'][:,1:].mean():.3f} "
              f"acc {out[f'{name}|rep|acc|{g}'][:,1:].mean():.3f} | "
              f"ctl NLL {out[f'{name}|ctl|nll|{g}'][:,1:].mean():.3f} "
              f"acc {out[f'{name}|ctl|acc|{g}'][:,1:].mean():.3f}")
    out[f"{name}|S"] = np.array(S)
    out[f"{name}|gaps"] = np.array(gaps)
    model.to("cpu")

(HERE / "cache").mkdir(exist_ok=True)
np.savez_compressed(HERE / "cache" / "copy_predict.npz", **out)
print("saved", HERE / "cache" / "copy_predict.npz")
