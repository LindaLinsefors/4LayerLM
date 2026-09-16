"""Pooled write-out (U) SIGNED cosine similarity — new A and sink decomposition C.

Write-side mirror of cos-sim/read-in/hide/compute_signed.py: pools the alive
components of the 8 matrices whose U factor WRITES to the residual stream --
h.<l>.attn.o_proj and h.<l>.mlp.down_proj -- and computes all pairwise signed
cosines between the unit write directions U_c (the write into the stream is
the rank-one a_c(x) * U_c).  Component sign is gauge, so each component's
(U, V) is first flipped to the majority-positive-activation convention of the
interactive cross widget (V_c.x > 0 on the majority of CI > 0.1 tokens;
comp_signs()) -- after that, cos(U_a, U_b) > 0 means the two components write
the same stream direction with the same polarity when active.

No gain-folding variant: unlike the read side (where the site RMSNorm gain
sits between the stream and V), writes go into the raw residual stream, so
raw U is exactly the write direction.

Alive: newA = cross_alive.npz ids (harvest-less sample-mean proxy), C =
sample mean CI > 1e-6 from coci_C.npz.  Saves to cache/ucos_signed_{newA,C}.npz:
  ids    (N,) int32   component id within its matrix
  site   (N,) int16   index into SITES
  hist   (800,) int64 signed-cos histogram, bins [-1,1]/800
  pairs  (P,3) float32 rows [i, j, cos] for |cos| > PAIR_MIN

Usage: python cos-sim/write-out/hide/compute_signed.py [newA C]  (default: both)
Runs on the default Python 3.11, ~1 min per decomposition.
"""
import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "coci-heatmaps" / "hide"))
from interactive_cross import comp_signs  # noqa: E402

CH = ROOT / "coci-heatmaps" / "hide" / "cache"
CD = ROOT / "compare-decomps" / "hide" / "cache"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

SITES = [f"h.{l}.{m}" for l in range(4) for m in ("attn.o_proj", "mlp.down_proj")]
PAIR_MIN = 0.4

DECOMPS = {
    "newA": dict(uv=CH / "uv_newA.npz", acts=CH / "act_signs_newA.npz"),
    "C": dict(uv=CD / "uv_C.npz", acts=CD / "act_signs_C.npz"),
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

    Us, ids, site_idx = [], [], []
    for s, site in enumerate(SITES):
        alive = alive_ids(name, site)
        sign = comp_signs(acts, site)[alive]                    # gauge fix
        Uall = uv[f"{site}|U"]
        # newA dump holds only the alive rows (in alive order); C holds all C
        U = (Uall if len(Uall) == len(alive) else Uall[alive]).astype(np.float32)
        Us.append(U * sign[:, None])
        ids.append(alive)
        site_idx.append(np.full(len(U), s, np.int16))
    U = np.concatenate(Us)
    ids = np.concatenate(ids)
    site_idx = np.concatenate(site_idx)
    N = len(U)
    print(f"{name}: {N} pooled components")

    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    C = (Un @ Un.T).astype(np.float32)
    iu = np.triu_indices(N, k=1)
    vals = C[iu]
    hist = np.histogram(vals, bins=800, range=(-1.0, 1.0))[0]
    m = np.abs(vals) > PAIR_MIN
    pairs = np.stack([iu[0][m], iu[1][m], vals[m]], axis=1).astype(np.float32)
    print(f"  median {np.median(vals):+.4f}, "
          f">{PAIR_MIN}: {(vals > PAIR_MIN).sum()}, "
          f">0.7: {(vals > 0.7).sum()}, >0.9: {(vals > 0.9).sum()}, "
          f"max {vals.max():.3f}; <-{PAIR_MIN}: {(vals < -PAIR_MIN).sum()}, "
          f"<-0.7: {(vals < -0.7).sum()}, min {vals.min():.3f}")
    np.savez_compressed(CACHE / f"ucos_signed_{name}.npz",
                        ids=ids, site=site_idx, hist=hist, pairs=pairs)
    print("  saved", CACHE / f"ucos_signed_{name}.npz")


for name in sys.argv[1:] or ["newA", "C"]:
    run(name)
