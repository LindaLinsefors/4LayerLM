"""Component activation strength |a_c| = ||U_c|| * |V_c . x| at the 17
heading-candidate positions (cache/hn_positions.npz), for all components of
all 24 decomposed matrices.  Pure forward pass on the target model + the
V/U factors — no CI network needed, runs locally in seconds.

Writes cache/act_cand.npz: per matrix `<m>|A` (C, 17) = |a_c| at candidate k
(same column order as hn_ci.npz CI_cand), plus cand_rc.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l  # noqa: E402

MATRICES = [f"h.{l}.{m}" for l in range(4)
            for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                      "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]

cand_rc = np.load(HERE / "cache/hn_positions.npz")["cand_rc"]
data_rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
uniq_rows = sorted({int(r) for r, _ in cand_rc})
batch = torch.stack([data_rows[r][:512] for r in uniq_rows])

model, pc, _ = load_pile_4l()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

inputs: dict[str, torch.Tensor] = {}
hooks = [model.get_submodule(m).register_forward_pre_hook(
    lambda mod, args, m=m: inputs.__setitem__(m, args[0].detach()))
    for m in MATRICES]
with torch.no_grad():
    model(batch.to(device))
for h in hooks:
    h.remove()

bi = [uniq_rows.index(int(r)) for r, _ in cand_rc]
pi = [int(p) for _, p in cand_rc]
data = {"cand_rc": cand_rc}
for m in MATRICES:
    comp = pc.components[m]
    V, U = comp.V.to(device), comp.U.to(device)          # (d_in, C), (C, d_out)
    x = inputs[m][bi, pi]                                # (17, d_in)
    a = (x @ V) * U.norm(dim=1)                          # (17, C)
    data[f"{m}|A"] = a.abs().T.cpu().numpy()
np.savez(HERE / "cache/act_cand.npz", **data)
print("saved cache/act_cand.npz;",
      {m: data[f"{m}|A"].shape for m in MATRICES[:2]})
