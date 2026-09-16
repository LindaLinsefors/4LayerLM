"""Count, for every pile_4l vocab token, how often it appears immediately
after <|endoftext|> in the training data (= how often it opens a document),
plus total token counts over the same data.

Data: the first N_SHARDS train shards of
danbraunai/pile-uncopyrighted-tok-shuffled (243,186 rows x 513 tokens each,
~204 MB each, downloaded to the HF cache and kept there). Only within-row
successors count (EOS as the last token of a row has no successor; rows are
shuffled, so cross-row pairs are meaningless).

Cache: endoftext-pos0/hide/cache/after_eos_counts.npz
  after_eos_counts (50277,) int64 — count of token right after an EOS
  total_counts     (50277,) int64 — count of token overall (same shards)
  n_rows, n_tokens, n_after_eos    — scan totals
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

N_SHARDS = 8
VOCAB = 50277
EOS_ID = 0

after_eos = np.zeros(VOCAB, dtype=np.int64)
total = np.zeros(VOCAB, dtype=np.int64)
n_rows = 0

for i in range(N_SHARDS):
    path = hf_hub_download("danbraunai/pile-uncopyrighted-tok-shuffled",
                           f"data/train-{i:05d}-of-02021.parquet", repo_type="dataset")
    col = pq.read_table(path, columns=["input_ids"])["input_ids"]
    flat = col.combine_chunks().flatten().to_numpy()
    a = flat.reshape(-1, 513)  # all rows are 513 tokens
    total += np.bincount(flat, minlength=VOCAB)
    after_eos += np.bincount(a[:, 1:][a[:, :-1] == EOS_ID], minlength=VOCAB)
    n_rows += len(a)
    print(f"shard {i}: {len(a):,} rows, running after-EOS samples "
          f"{after_eos.sum():,}, distinct {np.count_nonzero(after_eos):,}", flush=True)

np.savez(CACHE / "after_eos_counts.npz", after_eos_counts=after_eos,
         total_counts=total, n_rows=n_rows, n_tokens=n_rows * 513,
         n_after_eos=after_eos.sum())
print(f"done: {n_rows:,} rows, {n_rows * 513:,} tokens, "
      f"{after_eos.sum():,} after-EOS samples, "
      f"{np.count_nonzero(after_eos):,} distinct openers "
      f"-> {CACHE / 'after_eos_counts.npz'}")
