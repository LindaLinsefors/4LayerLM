"""Where is h.0.attn.q_proj:68 causally important on a real Q&A document?

Runs the decomposition's causal-importance function on cached Pile row 1795
(15 "A:" occurrences spread over positions 19..490) and prints the CI of
component 68 at every position where it is non-negligible, plus the CI at
every "A:" colon position.  Checks whether high CI tracks the "A:" pattern
or the end of the 512-token window.

Run with the 3.13 venv:
  param-decomp-vpd\\.venv\\Scripts\\python.exe pile-qk-comps/hide/ci_q68.py
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

MODULE = "h.0.attn.q_proj"
COMP = 68
ROW = 1795
CKPT = (ROOT / "prev_paper" / "models" / "pile_4layer"
        / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")

from param_decomp.models.component_model import ComponentModel

tok = load_tokenizer("pile_4l")
row = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")[ROW][:512]

model = ComponentModel.from_pretrained(str(CKPT))
model.eval()

with torch.no_grad():
    out = model(row[None, :], cache_type="input")
    ci = model.calc_causal_importances(out.cache, sampling="continuous")
ci68 = ci.lower_leaky[MODULE][0, :, COMP]          # (512,)

toks = [tok.convert_tokens_to_string([t]) for t in tok.convert_ids_to_tokens(row.tolist())]
a_colon = [i for i in range(1, 512)
           if toks[i] == ":" and toks[i - 1].rstrip().endswith("A")]

print(f"{MODULE}:{COMP} on pile row {ROW} (512 tokens)")
print(f'"A:" colon positions: {a_colon}')
print(f"\nCI at each \"A:\" colon (and the token after):")
for i in a_colon:
    print(f"  pos {i:3d}  {toks[i-1]!r}+{toks[i]!r} -> CI {ci68[i]:.4f}"
          f"   (next tok {toks[i+1]!r}: {ci68[i+1]:.4f})" if i + 1 < 512 else "")

thresh = 0.1
hot = (ci68 > thresh).nonzero().squeeze(-1).tolist()
print(f"\nAll positions with CI > {thresh} ({len(hot)}):")
for i in hot:
    ctx = "".join(toks[max(0, i - 4): i + 2]).replace("\n", "\\n")
    tag = " <-- A:" if i in a_colon else ""
    print(f"  pos {i:3d}  CI {ci68[i]:.3f}  ...{ctx!r}{tag}")

print(f"\nCI summary: max {ci68.max():.4f}, n>0.5: {(ci68 > 0.5).sum().item()}, "
      f"n>0.1: {(ci68 > 0.1).sum().item()}, n>0.01: {(ci68 > 0.01).sum().item()}")
print(f"mean CI in first half (0-255): {ci68[:256].mean():.5f}, "
      f"second half (256-511): {ci68[256:].mean():.5f}")
