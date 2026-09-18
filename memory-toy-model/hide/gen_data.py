"""Generate the nested fact dataset for the memory toy model.

Facts are random 3-token strings (t0, t1, t2) over vocab 1024:
  - all ordered input pairs (t0, t1) are unique  ->  unique correct completion
  - t2 is uniform random, independent (t2 collisions across facts are fine)
  - datasets are NESTED: the size-2^k dataset is the first 2^k rows of the
    same shuffled table, so "which facts got lost as n grew" is well-defined.

Output: hide/cache/facts_seed0.npz  { "facts": (2^18, 3) uint16 }
Deterministic: numpy default_rng(SEED). Runs locally in ~1 s.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 0
V = 1024
N_MAX = 2**18  # largest sweep size; smaller datasets are prefixes

rng = np.random.default_rng(SEED)

# Sample ordered pairs with margin, keep first occurrences in draw order.
# 2^20 draws from V^2 = 2^20 possible pairs leaves ~660k unique - plenty.
draws = rng.integers(0, V, size=(2**20, 2), dtype=np.int64)
keys = draws[:, 0] * V + draws[:, 1]
_, first_idx = np.unique(keys, return_index=True)
first_idx.sort()  # restore draw order -> prefix property = nesting
pairs = draws[first_idx[:N_MAX]]
assert len(pairs) == N_MAX, f"only {len(pairs)} unique pairs, raise the draw count"

t2 = rng.integers(0, V, size=N_MAX, dtype=np.int64)
facts = np.column_stack([pairs, t2]).astype(np.uint16)

out = HERE / "cache" / "facts_seed0.npz"
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, facts=facts)
print(f"wrote {out} shape={facts.shape} " f"unique pairs={len(np.unique(facts[:,0].astype(np.int64)*V + facts[:,1]))}")
