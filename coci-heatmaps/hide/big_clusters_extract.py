"""Extract all alive co-CI>0.9 clusters with >= MIN members from newA and newB
(same rule_clusters as the clustered09 reports) into cache/big_clusters.json:
{"newA": [{"site", "ids", "mean_ci"}, ...], "newB": [...]}, largest first.

Feeds big_clusters_modal.py / ab_cluster_report.py.

Usage: python coci-heatmaps/hide/big_clusters_extract.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from report import MATS  # noqa: E402
from report_clustered09 import rule_clusters  # noqa: E402

MIN = 5

out = {}
for name in ("newA", "newB"):
    stats = np.load(HERE / "cache" / f"coci_{name}.npz")
    clusters = []
    for l in range(4):
        for mat in MATS:
            mod = f"h.{l}.{mat}"
            r = stats[f"{mod}|r"].astype(np.float32)
            m = stats[f"{mod}|mean"]
            order_all = np.argsort(-m, kind="stable")
            alive = order_all[m[order_all] > 1e-6]
            multi, _ = rule_clusters(r, alive)
            for cl in multi:
                if len(cl) >= MIN:
                    cl = np.sort(cl)
                    clusters.append({"site": mod, "ids": cl.tolist(),
                                     "mean_ci": float(m[cl].mean())})
    clusters.sort(key=lambda c: -len(c["ids"]))
    out[name] = clusters
    print(f"{name}: {len(clusters)} clusters with >= {MIN} members, "
          f"{sum(len(c['ids']) for c in clusters)} comps total")

path = HERE / "cache" / "big_clusters.json"
path.write_text(json.dumps(out))
print("wrote", path)
