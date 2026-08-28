"""Harvest final-residual-stream activations (after ln_f, just before the
unembedding) for both models on representative training data.

Data: the cached training rows already used elsewhere in the project
(context-loss/hide/cache/): 200 Pile rows truncated to 512 tokens (n_ctx),
and the first 360 SimpleStories stories (~103k tokens) — every token position
is one activation vector.

Cache: token-embeds/hide/cache/final_acts_{pile_4l,simple_2l}.npz with
  acts       (N, d) float32 — the post-ln_f residual stream vectors
  acts_pre   (N, d) float32 — the same positions just BEFORE ln_f
  token_ids  (N,)   int32   — token at each position
  row        (N,)   int32   — which cached row/story the position came from
  pos        (N,)   int32   — position within that row
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l, load_simple_2l

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

N_PILE_ROWS = 200      # x 512 tokens = 102,400 vectors
N_SIMPLE_STORIES = 360  # ~103k tokens at mean story length 287

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def harvest(name: str, model, rows: list[torch.Tensor], batch_size: int) -> None:
    """Run `model` on `rows` (list of 1-D token-id tensors), capturing the
    output of ln_f at every position."""
    model = model.to(DEVICE)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    hook = model.ln_f.register_forward_hook(
        lambda _mod, inp, out: captured.append(
            (inp[0].detach().float().cpu(), out.detach().float().cpu())
        )
    )

    acts, acts_pre, token_ids, row_idx, pos = [], [], [], [], []
    with torch.no_grad():
        # group rows of equal length into batches (Pile rows all 512; stories vary)
        by_len: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by_len.setdefault(len(r), []).append(i)
        for length, idxs in sorted(by_len.items()):
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start : start + batch_size]
                batch = torch.stack([rows[i] for i in chunk]).to(DEVICE)
                captured.clear()
                model(batch)
                ((pre, out),) = captured  # each (B, T, d)
                acts.append(out.reshape(-1, out.shape[-1]))
                acts_pre.append(pre.reshape(-1, pre.shape[-1]))
                token_ids.append(batch.cpu().flatten())
                row_idx.append(torch.tensor(chunk).repeat_interleave(length))
                pos.append(torch.arange(length).repeat(len(chunk)))
    hook.remove()

    all_acts = torch.cat(acts)
    out_path = CACHE / f"final_acts_{name}.npz"
    np.savez(
        out_path,
        acts=all_acts.numpy(),
        acts_pre=torch.cat(acts_pre).numpy(),
        token_ids=torch.cat(token_ids).numpy().astype(np.int32),
        row=torch.cat(row_idx).numpy().astype(np.int32),
        pos=torch.cat(pos).numpy().astype(np.int32),
    )
    print(f"{name}: {all_acts.shape[0]:,} vectors (d={all_acts.shape[1]}) "
          f"from {len(rows)} rows -> {out_path}")


pile_rows = torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
)[:N_PILE_ROWS]
pile_rows = [r[:512] for r in pile_rows]  # rows are 513 tokens, n_ctx = 512

simple_rows = torch.load(
    ROOT / "context-loss/hide/cache/simple_stories.pt", map_location="cpu", weights_only=True
)[:N_SIMPLE_STORIES]
simple_rows = [r[:512] for r in simple_rows]  # a few stories exceed n_ctx = 512

model, _, _ = load_pile_4l()
harvest("pile_4l", model, pile_rows, batch_size=16)
del model

model, _, _ = load_simple_2l()
harvest("simple_2l", model, simple_rows, batch_size=64)
