"""Modal GPU job: cross-site co-CI statistics for the SIGNED write-out-cosine
clusters of new A and sink decomposition C (write-side mirror of
cos-sim/read-in/hide/cluster_ci_signed_modal.py; same statistics, member sets
from the U cosines).

Members: every component that appears in a pair with cos(U) > 0.7 (the
cluster-chaining threshold) OR cos(U) < -0.4 (so the negative-tail pairs get
co-CI values too), from cache/ucos_signed_{decomp}.npz.  Extracts their
per-token CI (clip(preact, 0, 1)) as one (T, K) series over the same 4,000
cached Pile rows and accumulates sum + Gram -> Pearson r between ANY two
members, same or different matrix.  Saved to
cache/cluster_ci_signed_{decomp}.npz: "site" (K, int16 index into SITES),
"ids" (K, int32), "S1"/"F" (K,), "G" (K, K), "T".

C decomposes the sink target t-87f91319, so its function runs on the
sink-loader PR #1002 image (commit 82a67f71c) against the shared
/data/sink-models root; newA runs on the pinned-commit image against its
per-archive root.  Both rely on archives already on volume vpd-4layer and
/pile_rows.npy.

Run (from the project root; Modal needs UTF-8 on this machine):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run cos-sim/write-out/hide/cluster_ci_signed_modal.py --decomp C
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run cos-sim/write-out/hide/cluster_ci_signed_modal.py --decomp newA
"""

import modal

app = modal.App("writeout-cluster-ci-signed")
vol = modal.Volume.from_name("vpd-4layer")

PINNED = "facf2e7b1d5273985ea1af277240f340f5bdeac0"   # public main (newA)
SINK = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"     # PR #1002 (C/D/E)


def make_image(commit):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "curl", "zstd")
        .pip_install(
            f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{commit}"
        )
    )


DECOMPS = {
    "newA": dict(run="p-8383f5e5", step=800000,
                 root="/data/new-decomps/p-8383f5e5-decomposition-800000"),
    "C": dict(run="p-d60af588", step=100000, root="/data/sink-models"),
}


def _compute(members, run, step, root_path):
    """members: {site_name: [comp ids]} -> S1/F/G over the concatenated columns."""
    import time
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")

    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run
    from jax.sharding import PartitionSpec as P

    FIRE_THRESH = 0.1
    root = Path(root_path)
    loaded = open_jax_run(root / "runs" / run, step=step, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    msites = [s for s in ci_fn.fn.output_names if s in members]
    idx = {s: jnp.asarray(np.asarray(members[s], np.int32)) for s in msites}
    K = sum(len(members[s]) for s in msites)
    print(f"{run}: {K} clustered components over {len(msites)} sites", flush=True)

    @eqx.filter_jit
    def step_fn(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        cols = [jnp.clip(pre[s], 0.0, 1.0).reshape(-1, pre[s].shape[-1])[:, idx[s]]
                for s in msites]
        c = jnp.concatenate(cols, axis=1)                     # (t, K)
        g = jnp.einsum("tc,td->cd", c, c, out_sharding=P(None, None))
        return c.sum(0), (c > FIRE_THRESH).sum(0), g

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    S1 = np.zeros(K, np.float64)
    F = np.zeros(K, np.float64)
    G = np.zeros((K, K), np.float64)
    T = 0
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            tok = jnp.asarray(rows[i:i + B])
            s1, f, g = step_fn(placed, ci_fn, tok)
            S1 += np.asarray(s1, np.float64)
            F += np.asarray(f, np.float64)
            G += np.asarray(g, np.float64)
            T += tok.size
            if (i // B) % 50 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"done, {T} tokens, {time.time() - t0:.0f}s", flush=True)
    return {"order": msites, "S1": S1, "F": F, "G": G, "T": T}


@app.function(image=make_image(PINNED), gpu="A10G", volumes={"/data": vol},
              timeout=3600)
def compute_pinned(members: dict, run: str, step: int, root_path: str) -> dict:
    return _compute(members, run, step, root_path)


@app.function(image=make_image(SINK), gpu="A10G", volumes={"/data": vol},
              timeout=3600)
def compute_sink(members: dict, run: str, step: int, root_path: str) -> dict:
    return _compute(members, run, step, root_path)


@app.local_entrypoint()
def main(decomp: str = "C") -> None:
    from pathlib import Path
    import numpy as np

    here = Path(__file__).resolve().parent
    SITES = [f"h.{l}.{m}" for l in range(4)
             for m in ("attn.o_proj", "mlp.down_proj")]
    cfg = DECOMPS[decomp]
    z = np.load(here / "cache" / f"ucos_signed_{decomp}.npz")
    P_ = z["pairs"]
    m = (P_[:, 2] > 0.7) | (P_[:, 2] < -0.4)
    pooled = np.unique(P_[m, :2].astype(int))
    site, ids = z["site"][pooled], z["ids"][pooled]
    members = {SITES[s]: ids[site == s].tolist() for s in np.unique(site)}
    print(f"{decomp}: {len(pooled)} clustered components")

    fn = compute_pinned if decomp == "newA" else compute_sink
    out = fn.remote(members, cfg["run"], cfg["step"], cfg["root"])
    # remote concat order = ci_fn output_names order; rebuild site/id vectors
    o_site = np.concatenate([np.full(len(members[s]), SITES.index(s), np.int16)
                             for s in out["order"]])
    o_ids = np.concatenate([np.asarray(members[s], np.int32)
                            for s in out["order"]])
    np.savez_compressed(here / "cache" / f"cluster_ci_signed_{decomp}.npz",
                        site=o_site, ids=o_ids, S1=out["S1"], F=out["F"],
                        G=out["G"], T=np.array(out["T"]))
    print("saved", here / "cache" / f"cluster_ci_signed_{decomp}.npz")
