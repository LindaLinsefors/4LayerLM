"""Sufficient statistics for co-activation / co-CI of all alive q/k components.

Runs 2000 cached Pile rows (1.02M tokens) through the ComponentModel on GPU
and accumulates, per matrix, over tokens t:
  - act:  S1[i] = sum_t |a_i(t)|,  G[i,j] = sum_t |a_i(t)||a_j(t)|
  - ci:   S1[i] = sum_t ci_i(t),   G[i,j] = sum_t ci_i(t) ci_j(t)
  - fire: F[i]  = sum_t [ci_i > 0.1],  FF[i,j] = sum_t [ci_i > 0.1][ci_j > 0.1]
for the alive components (harvest mean CI > 1e-6), where
a_c = ||U_c|| (V_c . phi) and ci = lower_leaky causal importance.
Pearson r and co-firing stats are then exact functions of these.

Output: cache/groups_stats.npz with keys "<mod>|{S1a,Ga,S1c,Gc,F,FF,idx}" + "T".

Run with the 3.13 venv (GPU):
  param-decomp-vpd\\.venv\\Scripts\\python.exe pile-qk-comps/hide/groups_compute.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

N_ROWS = 2000
BATCH = 16
FIRE_THRESH = 0.1
MATRICES = [f"h.{l}.attn.{m}" for l in range(4) for m in ("q_proj", "k_proj")]
CKPT = (ROOT / "prev_paper" / "models" / "pile_4layer"
        / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")
OUT_NPZ = HERE / "cache" / "groups_stats.npz"

from param_decomp.models.component_model import ComponentModel

device = "cuda" if torch.cuda.is_available() else "cpu"
alive = {m: torch.tensor(np.where(np.load(HERE / "cache" / "mean_ci.npz")[m] > 1e-6)[0])
         for m in MATRICES}

rows = torch.stack([r[:512] for r in
                    torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")[:N_ROWS]])

model = ComponentModel.from_pretrained(str(CKPT))
model.eval().to(device)
params = dict(model.named_parameters())
V = {m: params[f"_components.{m.replace('.', '-')}.V"].detach() for m in MATRICES}
u_norm = {m: params[f"_components.{m.replace('.', '-')}.U"].detach().norm(dim=1)
          for m in MATRICES}

acc = {m: {k: torch.zeros(len(alive[m]), device=device) for k in ("S1a", "S1c", "F")}
       | {k: torch.zeros(len(alive[m]), len(alive[m]), device=device)
          for k in ("Ga", "Gc", "FF")}
       for m in MATRICES}

t0 = time.time()
with torch.no_grad():
    for i in range(0, N_ROWS, BATCH):
        out = model(rows[i:i + BATCH], cache_type="input")
        ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
        for m in MATRICES:
            idx = alive[m].to(device)
            a = ((out.cache[m] @ V[m][:, idx]) * u_norm[m][idx]).reshape(-1, len(idx)).abs()
            c = ci_out.lower_leaky[m][:, :, idx].reshape(-1, len(idx))
            f = (c > FIRE_THRESH).float()
            acc[m]["S1a"] += a.sum(0)
            acc[m]["Ga"] += a.T @ a
            acc[m]["S1c"] += c.sum(0)
            acc[m]["Gc"] += c.T @ c
            acc[m]["F"] += f.sum(0)
            acc[m]["FF"] += f.T @ f
        if (i // BATCH) % 25 == 0:
            print(f"{i + BATCH}/{N_ROWS} rows, {time.time() - t0:.0f}s", flush=True)

data = {"T": np.array(N_ROWS * 512)}
for m in MATRICES:
    for k, v in acc[m].items():
        data[f"{m}|{k}"] = v.cpu().numpy()
    data[f"{m}|idx"] = alive[m].numpy()
np.savez(OUT_NPZ, **data)
print("saved", OUT_NPZ, f"({time.time() - t0:.0f}s total)")
