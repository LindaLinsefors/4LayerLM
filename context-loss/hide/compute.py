"""Per-token loss vs context length, for both models.

Fetches training-distribution data, runs the target models, and records for
every predicted token:
  loss  -- cross-entropy  -log p(target)  in nats
  (position t is the array column: the prediction used t+1 tokens of context)
  dist  -- within-document context length d: number of tokens since the last
           EOS/EOT in the context (d=0: predicting the first token of a new
           document; d=-1: the document started before the row, true context
           length unknown)
  target_is_eos -- the predicted token is the end-of-document token

Data is prepared exactly as in training (param_decomp.data.tokenize_and_concatenate):
  - pile_4l:   pre-tokenized 513-token rows, used as-is (<|endoftext|> id 0)
  - simple_2l: stories tokenized and concatenated with [EOS] (id 1) separators,
               chunked into 513-token rows

Each row of 513 tokens gives 512 predictions: inputs = row[:512],
targets = row[1:513].

Outputs (all in context-loss/):
  cache/pile_rows.pt, cache/simple_stories.pt  -- raw fetched data
  results/pile_4l.npz, results/simple_2l.npz   -- loss/dist/target_is_eos, (N, 512)

Runtime: a few minutes on the local GPU (most of it fetching data on the first run).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root, for load.py
import load

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROW_LEN = 513  # 512 inputs + 1 overhanging target
N_PILE_ROWS = 4000
N_STORIES = 6000
FETCH_CHUNK = 100  # datasets-server API max rows per request


def fetch_cached(cache_file: Path, fetch_fn, n: int) -> list[torch.Tensor]:
    """Fetch n samples in chunks of 100 via load.py, caching the result.
    Retries with backoff on rate limiting (HTTP 429) and saves partial progress."""
    import time
    import urllib.error

    rows = torch.load(cache_file, weights_only=True) if cache_file.exists() else []
    if len(rows) >= n:
        return rows[:n]
    for offset in range(len(rows), n, FETCH_CHUNK):
        for attempt in range(8):
            try:
                rows += fetch_fn(min(FETCH_CHUNK, n - offset), offset)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                wait = 15 * 2**attempt
                print(f"\n  rate limited at {len(rows)}, waiting {wait}s")
                torch.save(rows, cache_file)
                time.sleep(wait)
        print(f"  fetched {len(rows)}/{n}", end="\r", flush=True)
    print()
    torch.save(rows, cache_file)
    return rows


def pack_rows(stories: list[torch.Tensor], eos_id: int) -> torch.Tensor:
    """Concatenate stories separated by EOS and chunk into (N, ROW_LEN) rows,
    replicating the training pipeline's tokenize_and_concatenate."""
    eos = torch.tensor([eos_id])
    stream = torch.cat([t for s in stories for t in (s, eos)][:-1])
    n = len(stream) // ROW_LEN
    return stream[: n * ROW_LEN].reshape(n, ROW_LEN)


@torch.no_grad()
def per_token_losses(model, rows: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Cross-entropy in nats for each of the 512 predictions per row. (N, 512)"""
    model = model.to(DEVICE)
    out = []
    for i in range(0, len(rows), batch_size):
        b = rows[i : i + batch_size].to(DEVICE)
        logits = model(b[:, :-1])  # (B, 512, vocab)
        loss = F.cross_entropy(
            logits.flatten(0, 1), b[:, 1:].flatten(), reduction="none"
        )
        out.append(loss.reshape(b.shape[0], -1).cpu())
        print(f"  {min(i + batch_size, len(rows))}/{len(rows)} rows", end="\r")
    print()
    model.cpu()
    return torch.cat(out)


def doc_distance(inputs: torch.Tensor, eos_id: int) -> torch.Tensor:
    """d[n, t] = t - (position of last EOS in inputs[n, :t+1]), or -1 if none.
    d is the number of same-document tokens in the context when predicting
    target t (the EOS itself marks the document start and is not counted)."""
    idx = torch.arange(inputs.shape[1])
    last_eos = torch.where(inputs == eos_id, idx, -1).cummax(dim=1).values
    return torch.where(last_eos >= 0, idx - last_eos, -1)


def run_model(name: str, rows: torch.Tensor, eos_id: int, batch_size: int):
    if (RESULTS / f"{name}.npz").exists():
        print(f"{name}: results exist, skipping (delete results/{name}.npz to rerun)")
        return
    model, _, _ = load.load(name)
    loss = per_token_losses(model, rows, batch_size)
    dist = doc_distance(rows[:, :-1], eos_id)
    import numpy as np

    np.savez_compressed(
        RESULTS / f"{name}.npz",
        loss=loss.numpy().astype("float32"),
        dist=dist.numpy().astype("int16"),
        target_is_eos=(rows[:, 1:] == eos_id).numpy(),
    )
    print(f"{name}: {loss.shape[0]} rows, mean loss {loss.mean():.4f} nats, "
          f"{(dist >= 0).float().mean():.1%} of tokens have known doc start")


if __name__ == "__main__":
    print(f"device: {DEVICE}")

    print("Pile rows...")
    pile_rows = torch.stack(fetch_cached(CACHE / "pile_rows.pt", load.pile_samples, N_PILE_ROWS))
    run_model("pile_4l", pile_rows, eos_id=0, batch_size=8)

    print("SimpleStories...")
    stories = fetch_cached(CACHE / "simple_stories.pt", load.simple_samples, N_STORIES)
    print(f"  {len(stories)} stories, lengths: median {int(torch.tensor([len(s) for s in stories]).float().median())}, "
          f"max {max(len(s) for s in stories)}")
    story_rows = pack_rows(stories, eos_id=1)
    run_model("simple_2l", story_rows, eos_id=1, batch_size=64)
