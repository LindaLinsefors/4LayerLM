"""Which RoPE did the internal C/D/E harvest run with? Per-site Spearman between
harvest.sqlite mean CI (co-authors' 10M-token pipeline) and our 2.05M-token
sample mean CI computed under fitted vs configured RoPE (caches from
rope_decomp_check_modal.py). Also prints the fitted-vs-configured L0/alive
comparison. Local, ~1 min.

Result 2026-09-18: the harvest matches the CONFIGURED (textbook base-1e4)
forward -- mean Spearman C/D/E 0.962/0.967/0.975 configured vs
0.801/0.810/0.862 fitted, configured better on every one of the 72 sites.
"""
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CACHE = HERE / "cache"
RUNS = {"C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35"}
T = 4000 * 512
SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                   "mlp.c_fc", "mlp.down_proj")]

for name, run in RUNS.items():
    hdir = next((ROOT / "sink-models/runs" / run / "harvest").iterdir())
    con = sqlite3.connect(f"file:{hdir / 'harvest.sqlite'}?mode=ro", uri=True)
    fit = np.load(CACHE / f"mean_ci_fitted_{name}.npz")
    cfg = np.load(CACHE / f"mean_ci_configured_{name}.npz")
    print(f"== {name} ({run}) ==   Spearman(harvest mean CI, sample mean CI)")
    print(f"{'site':22s} {'configured':>10s} {'fitted':>10s} "
          f"{'L0>0.1 cfg':>10s} {'L0>0.1 fit':>10s}")
    rc, rf = [], []
    for site in SITES:
        rows = con.execute(
            "SELECT component_idx, mean_causal_importance FROM components "
            "WHERE site=? ORDER BY component_idx", (site,)).fetchall()
        h = np.zeros(max(r[0] for r in rows) + 1)
        for idx, m in rows:
            h[idx] = m
        rho_c = spearmanr(h, cfg[f"{site}|Sci"]).statistic
        rho_f = spearmanr(h, fit[f"{site}|Sci"]).statistic
        rc.append(rho_c)
        rf.append(rho_f)
        print(f"{site:22s} {rho_c:10.4f} {rho_f:10.4f} "
              f"{cfg[f'{site}|F'].sum() / T:10.3f} {fit[f'{site}|F'].sum() / T:10.3f}")
    con.close()
    for tag, d in [("configured", cfg), ("fitted", fit)]:
        alive = sum(int(((d[f"{s}|Sci"] / T) > 1e-6).sum()) for s in SITES)
        print(f"  {tag}: mean rho {np.mean(rc if tag == 'configured' else rf):.4f}, "
              f"min {np.min(rc if tag == 'configured' else rf):.4f}, "
              f"alive(>1e-6) {alive}")
    print()
