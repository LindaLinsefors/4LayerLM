"""Run the pile_4l model over Pile rows and record everything needed for the
capability analysis.

Reuses the cached rows fetched by context-loss/compute.py (400 of its 4000).
Each 513-token row gives 512 predictions: inputs = row[:512], targets = row[1:].

Output: results/predictions.npz with per-prediction arrays of shape (N, 512):
  loss     -- cross-entropy  -log p(target)  in nats                (float32)
  rank     -- 1-indexed rank of the true token in the model's dist  (int32)
  entropy  -- entropy of the predictive distribution in nats        (float32)
  top_ids  -- the model's top-10 token ids                          (N,512,10) int32
  top_p    -- their probabilities                                   (N,512,10) float16
plus rows (N, 513) int32 -- the token ids themselves, so downstream scripts
don't need the context-loss cache.

Runtime: ~1 min on the local GPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root, for load.py
import load

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 400
TOPK = 10
BATCH = 4  # logits are (B, 512, 50277) float32 -- keep the batch small


@torch.no_grad()
def main():
    rows = torch.stack(torch.load(HERE.parent / "context-loss" / "cache" / "pile_rows.pt",
                                  weights_only=True)[:N_ROWS])
    model, _, _ = load.load_pile_4l()
    model = model.to(DEVICE)

    out = {k: [] for k in ["loss", "rank", "entropy", "top_ids", "top_p"]}
    for i in range(0, N_ROWS, BATCH):
        b = rows[i : i + BATCH].to(DEVICE)
        logits = model(b[:, :-1])                       # (B, 512, V)
        logp = logits.log_softmax(-1)
        targets = b[:, 1:]
        logp_true = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out["loss"].append((-logp_true).cpu())
        out["rank"].append(((logp > logp_true.unsqueeze(-1)).sum(-1) + 1).cpu())
        out["entropy"].append((-(logp.exp() * logp).sum(-1)).cpu())
        top = logp.topk(TOPK, dim=-1)
        out["top_ids"].append(top.indices.cpu())
        out["top_p"].append(top.values.exp().cpu())
        print(f"  {min(i + BATCH, N_ROWS)}/{N_ROWS} rows", end="\r")
    print()

    np.savez_compressed(
        RESULTS / "predictions.npz",
        rows=rows.numpy().astype(np.int32),
        loss=torch.cat(out["loss"]).numpy().astype(np.float32),
        rank=torch.cat(out["rank"]).numpy().astype(np.int32),
        entropy=torch.cat(out["entropy"]).numpy().astype(np.float32),
        top_ids=torch.cat(out["top_ids"]).numpy().astype(np.int32),
        top_p=torch.cat(out["top_p"]).numpy().astype(np.float16),
    )
    loss = torch.cat(out["loss"])
    rank = torch.cat(out["rank"])
    print(f"{N_ROWS} rows: mean loss {loss.mean():.4f} nats, "
          f"top-1 acc {(rank == 1).float().mean():.1%}, "
          f"top-10 acc {(rank <= 10).float().mean():.1%}")


if __name__ == "__main__":
    main()
