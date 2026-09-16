"""Pooled read-in (V) cosine similarity for decomposition NEW A (p-8383f5e5).

Pools the alive components of the 16 matrices whose V factor reads the
residual stream -- h.<l>.attn.{q,k,v}_proj (input rms_1(h)) and
h.<l>.mlp.c_fc (input rms_2(h)) for l = 0..3; o_proj (reads attention-head
outputs) and down_proj (reads the MLP hidden layer) are excluded.

Computes the full |cos(V_a, V_b)| matrix over the pooled set (|.| because
component sign is gauge), both raw and with the site's RMSNorm gain folded
in (effective read on the unit-normalized stream is g (.) V, since
V.(x_hat (.) g) = (g (.) V).x_hat; gains differ per site so this matters for
cross-site comparison).  Saves to cache/vcos.npz:
  ids        (N,) int32   component id within its matrix
  site       (N,) int16   index into SITES
  hist_raw / hist_fold    (400,) int64  |cos| histogram, bins [0,1]/400, off-diag upper triangle
  pairs_raw / pairs_fold  (P,3) float32 rows [i, j, |cos|] for |cos| > PAIR_MIN
Inputs: coci-heatmaps/hide/cache/{uv_newA.npz, cross_alive.npz},
prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors.
Runs on the default Python 3.11, ~30 s.
"""
import numpy as np
from pathlib import Path
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CH = ROOT / "coci-heatmaps" / "hide" / "cache"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")]
PAIR_MIN = 0.4

uv = np.load(CH / "uv_newA.npz")
alive = np.load(CH / "cross_alive.npz")

st = safe_open(ROOT / "prev_paper" / "models" / "pile_4layer"
               / "target_model_t-9d2b8f02" / "model_step_99999.safetensors",
               framework="np")
gains = {}
for l in range(4):
    g1 = st.get_tensor(f"h.{l}.rms_1.weight").astype(np.float32)
    g2 = st.get_tensor(f"h.{l}.rms_2.weight").astype(np.float32)
    for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj"):
        gains[f"h.{l}.{m}"] = g1
    gains[f"h.{l}.mlp.c_fc"] = g2

Vs, ids, site_idx = [], [], []
for s, site in enumerate(SITES):
    V = uv[f"{site}|V"].astype(np.float32)          # (n_alive, 768)
    Vs.append(V)
    ids.append(alive[f"newA|{site}"].astype(np.int32))
    site_idx.append(np.full(len(V), s, np.int16))
V = np.concatenate(Vs)
ids = np.concatenate(ids)
site_idx = np.concatenate(site_idx)
N = len(V)
print(f"{N} pooled components")

g = np.stack([gains[SITES[s]] for s in site_idx])    # (N, 768) per-row gain
out = {"ids": ids, "site": site_idx}
for tag, X in (("raw", V), ("fold", V * g)):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = np.abs(Xn @ Xn.T).astype(np.float32)
    iu = np.triu_indices(N, k=1)
    vals = C[iu]
    out[f"hist_{tag}"] = np.histogram(vals, bins=400, range=(0.0, 1.0))[0]
    m = vals > PAIR_MIN
    out[f"pairs_{tag}"] = np.stack(
        [iu[0][m], iu[1][m], vals[m]], axis=1).astype(np.float32)
    print(f"{tag}: median {np.median(vals):.4f}, "
          f">{PAIR_MIN}: {m.sum()}, >0.7: {(vals > 0.7).sum()}, "
          f">0.9: {(vals > 0.9).sum()}, max {vals.max():.3f}")
    del C, vals

np.savez_compressed(CACHE / "vcos.npz", **out)
print("saved", CACHE / "vcos.npz")
