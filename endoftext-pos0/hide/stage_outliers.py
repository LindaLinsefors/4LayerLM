"""Who are the extreme-norm positions per stage in resid_stage_norms.png?

Prints, for each residual-stream stage, the max / p99.9 / p99 / median norm and
the top-5 positions with their token and preceding context. Same 1,000 cached
Pile rows and stage capture as resid_stage_norms.py.

Finding (2026-08-28): the far outliers (~250 after MLPs 2-3) are the position-0
tokens of every row, whatever the token — the first-token attention-sink site.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS, BATCH_SIZE = 1000, 16

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:512] for r in rows]

model, _, tok = load_pile_4l()
model = model.to(DEVICE)
n_layer = len(model.h)
stages = (["after embedding"]
          + [f"after {kind} {l + 1}" for l in range(n_layer) for kind in ("attention", "MLP")]
          + ["after final norm"])

caps, hooks = {}, []
for l, block in enumerate(model.h):
    hooks.append(block.attn.register_forward_hook(
        lambda _m, _i, out, l=l: caps.__setitem__(f"attn{l}", out)))
    hooks.append(block.register_forward_hook(
        lambda _m, _i, out, l=l: caps.__setitem__(f"block{l}", out)))
hooks.append(model.ln_f.register_forward_hook(
    lambda _m, _i, out: caps.__setitem__("ln_f", out)))

norms = [[] for _ in stages]
with torch.no_grad():
    for start in range(0, len(rows), BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        hs = [model.wte(batch)]
        for l in range(n_layer):
            hs.append(hs[-1] + caps[f"attn{l}"])
            hs.append(caps[f"block{l}"])
        hs.append(caps["ln_f"])
        for si, h in enumerate(hs):
            norms[si].append(h.float().norm(dim=-1).flatten().cpu())
norms = np.stack([torch.cat(n).numpy() for n in norms])
ids = torch.cat(rows).numpy()

for stage, n in zip(stages, norms):
    top = np.argsort(n)[::-1][:5]
    print(f"\n{stage}:  max {n.max():.1f}, p99.9 {np.percentile(n, 99.9):.1f}, "
          f"p99 {np.percentile(n, 99):.1f}, median {np.median(n):.1f}")
    for i in top:
        r, p = divmod(int(i), 512)
        ctx = tok.decode(ids[i - min(6, p): i].tolist()) if p else ""
        t = tok.decode([int(ids[i])])
        print(f"   norm {n[i]:7.1f}  row {r:4d} pos {p:3d}  token {t!r}  after {ctx!r}")
