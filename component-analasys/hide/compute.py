"""Compute per-subcomponent magnitudes ||V_c|| * ||U_c|| for the Pile 4-layer
decomposition and classify each component by activity:

  alive  -- mean causal importance > 1e-6 (paper definition), from harvest.db
  semi   -- in harvest.db (fired at least once at threshold 0) but not alive
  never  -- not in harvest.db at all (never fired in 20,000 x 32 sequences)

Caches everything plot.py needs into cache/magnitudes.npz (fast to re-plot).
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from load import PILE_4L, ParameterComponents

CACHE = Path(__file__).parent / "cache" / "magnitudes.npz"
HARVEST_DB = PILE_4L / "additional-component-data" / "harvest.db"

ALIVE, SEMI, NEVER = 0, 1, 2  # category codes stored in the cache


def main():
    print("loading decomposition checkpoint...")
    pc = ParameterComponents(PILE_4L / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")

    # mean causal importance per component_key, for components that ever fired
    print("reading mean CI from harvest.db...")
    con = sqlite3.connect(f"file:{HARVEST_DB}?mode=ro", uri=True)
    mean_ci = {
        key: json.loads(blob)["causal_importance"]
        for key, blob in con.execute("SELECT component_key, mean_activations FROM components")
    }
    con.close()

    modules, indices, magnitudes, categories = [], [], [], []
    for module, sub in pc.components.items():  # V (d_in, C), U (C, d_out)
        mags = torch.linalg.norm(sub.V, dim=0) * torch.linalg.norm(sub.U, dim=1)
        for c, mag in enumerate(mags.tolist()):
            ci = mean_ci.get(f"{module}:{c}")
            cat = NEVER if ci is None else (ALIVE if ci > 1e-6 else SEMI)
            modules.append(module)
            indices.append(c)
            magnitudes.append(mag)
            categories.append(cat)

    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE,
        module=np.array(modules),
        index=np.array(indices),
        magnitude=np.array(magnitudes),
        category=np.array(categories),
    )

    mag = np.array(magnitudes)
    cat = np.array(categories)
    print(f"cached {len(mag)} components -> {CACHE}")
    for code, name in [(ALIVE, "alive"), (SEMI, "semi"), (NEVER, "never")]:
        m = mag[cat == code]
        print(
            f"  {name:>5}: n={len(m):>6}  magnitude min={m.min():.3g} "
            f"median={np.median(m):.3g} max={m.max():.3g}"
        )


if __name__ == "__main__":
    main()
