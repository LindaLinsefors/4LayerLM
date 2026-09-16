"""Modal GPU job: cross-site co-CI statistics for the newA clustered components.

The read-in cosine clusters (newA_report.md) span different matrices, but the
per-matrix cache coci_newA.npz only holds within-matrix correlations.  This
job extracts the per-token CI (clip(preact, 0, 1), same as coci_compute_new_modal)
of ALL clustered components — every component that appears in a |cos(V)| > 0.7
pair, i.e. exactly the union of the report's clusters — as one (T, K) series
over the same 4,000 cached Pile rows, and accumulates sum + Gram, so Pearson r
between ANY two clustered components (same or different matrix) can be formed.

The member list is computed locally from cache/vcos.npz and passed as an
argument; results come back over the wire (K ~ 1e3, Gram ~ 8 MB) and are
saved to cache/cluster_ci.npz: "site" (K, int16 index into SITES), "ids"
(K, int32), "S1"/"F" (K,), "G" (K, K), "T".

Run (from the project root; Modal needs UTF-8 on this machine):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run cos-sim/read-in/hide/cluster_ci_modal.py
Requires the decomposition archives already on the volume (they are, from
coci-heatmaps/hide/coci_compute_new_modal.py's fetch) and /pile_rows.npy.
"""

import modal

app = modal.App("newA-cluster-ci")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUN = "p-8383f5e5"  # newA


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(members: dict) -> dict:
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
    root = Path(f"/data/new-decomps/{RUN}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / RUN, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    msites = [s for s in ci_fn.fn.output_names if s in members]
    idx = {s: jnp.asarray(np.asarray(members[s], np.int32)) for s in msites}
    K = sum(len(members[s]) for s in msites)
    print(f"{RUN}: {K} clustered components over {len(msites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
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
            s1, f, g = step(placed, ci_fn, tok)
            S1 += np.asarray(s1, np.float64)
            F += np.asarray(f, np.float64)
            G += np.asarray(g, np.float64)
            T += tok.size
            if (i // B) % 50 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"done, {T} tokens, {time.time() - t0:.0f}s", flush=True)
    return {"order": msites, "S1": S1, "F": F, "G": G, "T": T}


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path
    import numpy as np

    here = Path(__file__).resolve().parent
    SITES = [f"h.{l}.{m}" for l in range(4)
             for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")]
    z = np.load(here / "cache" / "vcos.npz")
    P_ = z["pairs_raw"]
    m = P_[:, 2] > 0.7
    pooled = np.unique(P_[m, :2].astype(int))     # all clustered components
    site, ids = z["site"][pooled], z["ids"][pooled]
    members = {SITES[s]: ids[site == s].tolist() for s in np.unique(site)}
    print(f"{len(pooled)} clustered components")

    out = compute.remote(members)
    # remote concat order = ci_fn output_names order; rebuild the site/id vectors
    o_site = np.concatenate([np.full(len(members[s]), SITES.index(s), np.int16)
                             for s in out["order"]])
    o_ids = np.concatenate([np.asarray(members[s], np.int32)
                            for s in out["order"]])
    np.savez_compressed(here / "cache" / "cluster_ci.npz",
                        site=o_site, ids=o_ids, S1=out["S1"], F=out["F"],
                        G=out["G"], T=np.array(out["T"]))
    print("saved", here / "cache" / "cluster_ci.npz")
