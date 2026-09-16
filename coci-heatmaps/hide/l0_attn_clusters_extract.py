"""Extract ALL alive co-CI>0.9 clusters with >= 2 members of the four
h.0.attn matrices (q/k/v/o_proj), for the old decomposition (s-55ea3f9b) and
the two new ones (newA/newB), into cache/l0_attn_clusters.json:
{"old": [{"site", "ids", "mean_ci"}, ...], "newA": [...], "newB": [...]}.

Same rule_clusters as the clustered09 reports; alive/mean CI = harvest mean CI
for old, sample mean CI for newA/newB (no harvest DB).

Feeds l0_attn_clusters_modal.py / l0_attn_cluster_report.py.

Usage: python coci-heatmaps/hide/l0_attn_clusters_extract.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from report_clustered09 import rule_clusters  # noqa: E402

MIN = 2
SITES = [f"h.0.attn.{m}_proj" for m in "qkvo"]

out = {}
for name in ("old", "newA", "newB"):
    stats = np.load(HERE / "cache" /
                    ("coci_pile_4l.npz" if name == "old" else f"coci_{name}.npz"))
    mci = (np.load(HERE / "cache" / "mean_ci_pile_4l.npz") if name == "old"
           else {s: stats[f"{s}|mean"] for s in SITES})
    clusters = []
    for mod in SITES:
        r = stats[f"{mod}|r"].astype(np.float32)
        m = mci[mod]
        order_all = np.argsort(-m, kind="stable")
        alive = order_all[m[order_all] > 1e-6]
        multi, _ = rule_clusters(r, alive)
        for cl in multi:
            if len(cl) >= MIN:
                cl = np.sort(cl)
                clusters.append({"site": mod, "ids": cl.tolist(),
                                 "mean_ci": float(m[cl].mean())})
    out[name] = clusters
    per_site = ", ".join(
        f"{s}: {sum(1 for c in clusters if c['site'] == s)}" for s in SITES)
    print(f"{name}: {len(clusters)} clusters, "
          f"{sum(len(c['ids']) for c in clusters)} comps ({per_site})")

path = HERE / "cache" / "l0_attn_clusters.json"
path.write_text(json.dumps(out))
print("wrote", path)
