"""Pooled read-in (V) SIGNED cosine similarity — new A and sink decomposition C.

Same pooling as compute.py (the unsigned new-A analysis) but cosines keep
their sign (user request 2026-09-08).  Component sign is gauge, so each
component's (U, V) is first flipped to the majority-positive-activation
convention of the interactive cross widget (V_c.x > 0 on the majority of
CI > 0.1 tokens; comp_signs() from coci-heatmaps/hide/interactive_cross.py) —
after that, cos(V_a, V_b) > 0 means the two components read the same feature
with the same input polarity.

Pools the alive components of the 16 matrices whose V factor reads the
residual stream -- h.<l>.attn.{q,k,v}_proj (input rms_1(h)) and
h.<l>.mlp.c_fc (input rms_2(h)); o_proj/down_proj excluded.  Alive:
newA = cross_alive.npz ids (harvest-less sample-mean proxy), C = sample mean
CI > 1e-6 from coci_C.npz.  C decomposes the sink target t-87f91319 (untied
head — irrelevant here, V reads the input side).  Saves to
cache/vcos_signed_{newA,C}.npz:
  ids        (N,) int32   component id within its matrix
  site       (N,) int16   index into SITES
  hist_raw / hist_fold    (800,) int64  signed-cos histogram, bins [-1,1]/800
  pairs_raw / pairs_fold  (P,3) float32 rows [i, j, cos] for |cos| > PAIR_MIN

Usage: python cos-sim/read-in/hide/compute_signed.py [newA C]   (default: both)
Runs on the default Python 3.11, ~1 min per decomposition.
"""
import sys
import numpy as np
from pathlib import Path
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "coci-heatmaps" / "hide"))
from interactive_cross import comp_signs  # noqa: E402

CH = ROOT / "coci-heatmaps" / "hide" / "cache"
CD = ROOT / "compare-decomps" / "hide" / "cache"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")]
PAIR_MIN = 0.4

DECOMPS = {
    "newA": dict(
        uv=CH / "uv_newA.npz", acts=CH / "act_signs_newA.npz",
        st=ROOT / "prev_paper" / "models" / "pile_4layer"
        / "target_model_t-9d2b8f02" / "model_step_99999.safetensors"),
    "C": dict(
        uv=CD / "uv_C.npz", acts=CD / "act_signs_C.npz",
        st=ROOT / "sink-models" / "pretrain_cache" / "spd-t-87f91319"
        / "model_step_100000.safetensors"),
}


def alive_ids(name, site):
    if name == "newA":  # uv_newA rows are exactly these ids, in this order
        return np.load(CH / "cross_alive.npz")[f"newA|{site}"].astype(np.int32)
    cc = np.load(CH / f"coci_{name}.npz")
    return np.flatnonzero(cc[f"{site}|mean"] > 1e-6).astype(np.int32)


def run(name):
    cfg = DECOMPS[name]
    uv = np.load(cfg["uv"])
    acts = np.load(cfg["acts"])
    st = safe_open(cfg["st"], framework="np")
    gains = {}
    for l in range(4):
        g1 = st.get_tensor(f"h.{l}.rms_1.weight").astype(np.float32)
        g2 = st.get_tensor(f"h.{l}.rms_2.weight").astype(np.float32)
        for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj"):
            gains[f"h.{l}.{m}"] = g1
        gains[f"h.{l}.mlp.c_fc"] = g2

    Vs, ids, site_idx = [], [], []
    for s, site in enumerate(SITES):
        alive = alive_ids(name, site)
        sign = comp_signs(acts, site)[alive]                    # gauge fix
        Vall = uv[f"{site}|V"]
        # newA dump holds only the alive rows (in alive order); C holds all C
        V = (Vall if len(Vall) == len(alive) else Vall[alive]).astype(np.float32)
        Vs.append(V * sign[:, None])
        ids.append(alive)
        site_idx.append(np.full(len(V), s, np.int16))
    V = np.concatenate(Vs)
    ids = np.concatenate(ids)
    site_idx = np.concatenate(site_idx)
    N = len(V)
    print(f"{name}: {N} pooled components")

    g = np.stack([gains[SITES[s]] for s in site_idx])   # (N, 768) per-row gain
    out = {"ids": ids, "site": site_idx}
    for tag, X in (("raw", V), ("fold", V * g)):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        C = (Xn @ Xn.T).astype(np.float32)
        iu = np.triu_indices(N, k=1)
        vals = C[iu]
        out[f"hist_{tag}"] = np.histogram(vals, bins=800, range=(-1.0, 1.0))[0]
        m = np.abs(vals) > PAIR_MIN
        out[f"pairs_{tag}"] = np.stack(
            [iu[0][m], iu[1][m], vals[m]], axis=1).astype(np.float32)
        print(f"  {tag}: median {np.median(vals):+.4f}, "
              f">{PAIR_MIN}: {(vals > PAIR_MIN).sum()}, "
              f">0.7: {(vals > 0.7).sum()}, >0.9: {(vals > 0.9).sum()}, "
              f"max {vals.max():.3f}; <-{PAIR_MIN}: {(vals < -PAIR_MIN).sum()}, "
              f"<-0.7: {(vals < -0.7).sum()}, min {vals.min():.3f}")
        del C, vals
    np.savez_compressed(CACHE / f"vcos_signed_{name}.npz", **out)
    print("  saved", CACHE / f"vcos_signed_{name}.npz")


for name in sys.argv[1:] or ["newA", "C"]:
    run(name)
