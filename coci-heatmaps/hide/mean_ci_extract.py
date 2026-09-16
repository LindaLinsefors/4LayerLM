"""Extract harvest-DB mean CI for ALL components of every matrix, both models.

Writes hide/cache/mean_ci_<model>.npz: one array per module name (e.g.
"h.0.mlp.c_fc"), full length C, harvest mean CI (0.0 for components absent
from the harvest = never fired). Used for the alive cut (mean CI > 1e-6,
project convention) and for sorting heatmap axes by descending mean CI.

Pile: mean CI is json_extract(mean_activations, '$.causal_importance').
Simple: older schema, mean_ci REAL column directly.

Run: python coci-heatmaps/hide/mean_ci_extract.py   (~1 min, mostly pile DB I/O)
"""

import sqlite3
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

C_PER_MODULE = {
    "pile_4l": {"attn.q_proj": 512, "attn.k_proj": 512, "attn.v_proj": 1024,
                "attn.o_proj": 1024, "mlp.c_fc": 3072, "mlp.down_proj": 3584},
    "simple_2l": {"attn.q_proj": 288, "attn.k_proj": 288, "attn.v_proj": 384,
                  "attn.o_proj": 480, "mlp.c_fc": 1152, "mlp.down_proj": 960},
}
N_LAYERS = {"pile_4l": 4, "simple_2l": 2}
DB = {
    "pile_4l": ROOT / "prev_paper/models/pile_4layer/additional-component-data/harvest.db",
    "simple_2l": ROOT / "prev_paper/models/simplestories_2layer/additional-component-data"
                        "/harvest/s-eab2ace8/h-20260212_142914/harvest.db",
}
QUERY = {
    "pile_4l": "SELECT component_key, json_extract(mean_activations,"
               " '$.causal_importance') FROM components",
    "simple_2l": "SELECT component_key, mean_ci FROM components",
}

for model in ("pile_4l", "simple_2l"):
    arrays = {f"h.{l}.{m}": np.zeros(c)
              for l in range(N_LAYERS[model])
              for m, c in C_PER_MODULE[model].items()}
    con = sqlite3.connect(f"file:{DB[model]}?mode=ro", uri=True)
    n = 0
    for key, ci in con.execute(QUERY[model]):
        mod, idx = key.rsplit(":", 1)
        arrays[mod][int(idx)] = ci
        n += 1
    con.close()
    alive = sum(int((a > 1e-6).sum()) for a in arrays.values())
    print(f"{model}: {n} harvest rows, {alive} alive (mean CI > 1e-6)")
    np.savez(CACHE / f"mean_ci_{model}.npz", **arrays)
