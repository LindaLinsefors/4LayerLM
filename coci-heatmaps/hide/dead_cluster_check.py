"""What do the 38 dead h.3.attn.v_proj components (the big all_clustered09
cluster) fire on? Pulls each member's harvest.db activating examples, finds
the max-CI position per example, and tallies trigger tokens + contexts.

Usage: python coci-heatmaps/hide/dead_cluster_check.py
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

MEMBERS = [5, 14, 56, 108, 168, 174, 201, 221, 227, 243, 264, 289, 326, 343,
           363, 367, 387, 415, 430, 474, 476, 498, 507, 564, 567, 569, 617,
           644, 654, 697, 718, 736, 739, 770, 820, 876, 940, 971]
MOD = "h.3.attn.v_proj"
DATA = ROOT / "prev_paper/models/pile_4layer/additional-component-data"

tok = load_tokenizer("pile_4l")
hdb = sqlite3.connect(f"file:{DATA / 'harvest.db'}?mode=ro", uri=True)
idb = sqlite3.connect(f"file:{DATA / 'interp.db'}?mode=ro", uri=True)

trigger_tally = Counter()
context_shown = 0
for c in MEMBERS:
    key = f"{MOD}:{c}"
    row = hdb.execute(
        "SELECT firing_density, n_activation_examples, activation_examples "
        "FROM components WHERE component_key = ?", (key,)).fetchone()
    lab = idb.execute("SELECT label, confidence FROM interpretations "
                      "WHERE component_key = ?", (key,)).fetchone()
    if row is None:
        print(f"{key}: not in harvest.db")
        continue
    dens, n_ex, ex_json = row
    examples = json.loads(ex_json)
    triggers = Counter()
    peaks = []
    for ex in examples:
        ci = ex["activations"]["causal_importance"]
        p = max(range(len(ci)), key=lambda i: ci[i])
        triggers[ex["token_ids"][p]] += 1
        peaks.append((ci[p], ex["token_ids"], p))
    trigger_tally.update(triggers)
    top = ", ".join(f"{tok.decode([t])!r}x{n}" for t, n in triggers.most_common(4))
    print(f"{key}: density {dens:.2e}, {len(examples)} examples; "
          f"label={lab[0] if lab else '(none)'} [{lab[1] if lab else '-'}]")
    print(f"    max-CI trigger tokens: {top}")
    if context_shown < 6:  # a few full contexts, highest-CI examples
        peaks.sort(reverse=True)
        for ci_v, ids, p in peaks[:2]:
            lo = max(0, p - 12)
            ctx = tok.decode(ids[lo:p]) + "[[" + tok.decode([ids[p]]) + "]]" \
                + tok.decode(ids[p + 1:p + 6])
            print(f"    CI {ci_v:.4f}: ...{ctx!r}")
        context_shown += 1

print("\n=== trigger-token tally over all members' examples ===")
for t, n in trigger_tally.most_common(15):
    print(f"{n:6d}  {t:6d}  {tok.decode([t])!r}")
