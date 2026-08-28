"""Max causal importance ever reached by each component of one activity class
("alive" or "semi" = ever-active-but-not-alive) of the Pile 4-layer
decomposition:  python compute_max_ci.py <class>

harvest.db stores no max-CI column, but each component row carries per-token CI
traces for its stored activation examples (reservoir-sampled from all firings,
capped at 1000 examples / 5 per batch). The max over those traces is exact for
components with few firings and a lower bound otherwise. Cached to
cache/max_ci_<class>.npz.

Runtime is dominated by reading + JSON-parsing the examples (tens of GB).
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

try:
    import orjson as jsonlib
except ImportError:
    import json as jsonlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from load import PILE_4L

HARVEST_DB = PILE_4L / "additional-component-data" / "harvest.db"
CATEGORY_CODES = {"alive": 0, "semi": 1}


def main(cls: str):
    cache = Path(__file__).parent / "cache" / f"max_ci_{cls}.npz"
    mag = np.load(Path(__file__).parent / "cache" / "magnitudes.npz")
    sel = mag["category"] == CATEGORY_CODES[cls]
    keys = [f"{m}:{i}" for m, i in zip(mag["module"][sel], mag["index"][sel])]
    print(f"{len(keys)} {cls} components")

    con = sqlite3.connect(f"file:{HARVEST_DB}?mode=ro", uri=True)
    con.execute("CREATE TEMP TABLE keys(k TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO keys VALUES (?)", [(k,) for k in keys])

    max_ci, max_abs_act, n_examples = {}, {}, {}
    t0 = time.time()
    rows = con.execute(
        "SELECT c.component_key, c.n_activation_examples, c.activation_examples"
        " FROM components c JOIN keys ON c.component_key = keys.k"
    )
    for n_done, (key, n_ex, blob) in enumerate(rows, 1):
        examples = jsonlib.loads(blob)
        max_ci[key] = max(max(ex["activations"]["causal_importance"]) for ex in examples)
        max_abs_act[key] = max(
            max(abs(a) for a in ex["activations"]["component_activation"]) for ex in examples
        )
        n_examples[key] = n_ex
        if n_done % 2000 == 0:
            print(f"  {n_done}/{len(keys)}  ({time.time() - t0:.0f} s)")
    con.close()
    assert len(max_ci) == len(keys), f"only found {len(max_ci)} of {len(keys)}"

    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        cache,
        module=mag["module"][sel],
        index=mag["index"][sel],
        max_ci=np.array([max_ci[k] for k in keys]),
        max_abs_act=np.array([max_abs_act[k] for k in keys]),
        n_examples=np.array([n_examples[k] for k in keys]),
    )
    v = np.array([max_ci[k] for k in keys])
    print(f"cached {len(v)} components -> {cache}")
    print(
        f"max CI percentiles: 1%={np.percentile(v, 1):.3g} 50%={np.percentile(v, 50):.3g} "
        f"99%={np.percentile(v, 99):.3g} max={v.max():.3g}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
