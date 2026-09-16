"""Position sweep for h.0.attn.q_proj:68 ("\\n\\nA:\\n\\n" StackExchange answer marker).

Takes a real firing context from harvest.db (41-token window, colon at index
20, CI = 1.0 there) and embeds it into a 512-token sequence at varying
absolute positions (filler: cached Pile row 0, which contains no "A:").
Evaluates the causal-importance function at each placement: if CI were
position-dependent (e.g. only high near the end of the training chunk), the
sweep would show it.

Run with the 3.13 venv:
  prev_paper\\param-decomp-vpd\\.venv\\Scripts\\python.exe pile-qk-comps/hide/ci_q68_sweep.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

MODULE = "h.0.attn.q_proj"
COMP = 68
CKPT = (ROOT / "prev_paper" / "models" / "pile_4layer"
        / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")
DB = (ROOT / "prev_paper" / "models" / "pile_4layer"
      / "additional-component-data" / "harvest.db")

from param_decomp.models.component_model import ComponentModel

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
exs = json.loads(con.execute(
    "SELECT activation_examples FROM components WHERE component_key=?",
    (f"{MODULE}:{COMP}",)).fetchone()[0])
ex = exs[0]                                     # colon at index 20, CI 1.0
ids = ex["token_ids"]
colon_in_window = ex["activations"]["causal_importance"].index(
    max(ex["activations"]["causal_importance"]))
window = torch.tensor(ids)
left = colon_in_window                          # tokens before the colon in the window

filler = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")[0][:512]

model = ComponentModel.from_pretrained(str(CKPT))
model.eval()

positions = [21, 40, 80, 160, 256, 350, 450, 490, 508]
seqs, colons = [], []
for p in positions:
    n_pre = p - left                            # filler tokens before the window
    seq = torch.cat([filler[:n_pre], window])[:512]
    seq = torch.cat([seq, filler[n_pre:n_pre + 512 - len(seq)]])
    assert len(seq) == 512 and seq[p] == window[colon_in_window]
    seqs.append(seq)
    colons.append(p)

with torch.no_grad():
    out = model(torch.stack(seqs), cache_type="input")
    ci = model.calc_causal_importances(out.cache, sampling="continuous")
ci68 = ci.lower_leaky[MODULE][:, :, COMP]       # (n_positions, 512)

print(f"{MODULE}:{COMP} — CI at the \"A:\" colon vs absolute position (n_ctx = 512):")
for b, p in enumerate(colons):
    others = ci68[b].clone()
    others[p] = 0
    print(f"  colon at pos {p:3d}: CI = {ci68[b, p]:.4f}   "
          f"(max CI elsewhere in seq: {others.max():.4f})")
