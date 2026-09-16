"""Modal GPU job: top activating tokens per component for the two NEW 800k-step
JAX decompositions (newA = p-8383f5e5, newB = p-4d9a6a12) of the pile_4l target.

For every component of every matrix, accumulates summed CI per *input token id*
(CI = clip(preactivations, 0, 1), same as coci_compute_new_modal.py) over the
4,000 cached Pile rows (2.05M tokens), then keeps the top-K token ids. These
stand in for autointerp labels in the mean-CI widget: the new runs have no
harvest/autointerp data, only Goodfire-internal harvests.

Setup (decomposition archives + /pile_rows.npy in volume vpd-4layer) is the one
made by coci-heatmaps/hide/coci_compute_new_modal.py — run that first if the
volume is empty. Forward+CI runs under the run's mesh; the (vocab, C)
scatter-add accumulators live on GPU outside the mesh context (CI batches
roundtrip through host as float16).

Run:  modal run mean-ci-widget/hide/top_tokens_compute_modal.py
Then: modal volume get vpd-4layer /top_tokens_newA.npz mean-ci-widget/hide/cache/
      modal volume get vpd-4layer /top_tokens_newB.npz mean-ci-widget/hide/cache/
Output npz keys per module: "<mod>|top_ids" (C, K) int32, "<mod>|top_ci" (C, K)
float32 (summed CI, descending), "<mod>|total" (C,) float64 (total summed CI).
"""

import modal

app = modal.App("top-tokens-new-decomps")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
VOCAB = 50277
K = 20


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(name: str) -> None:
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

    run = RUNS[name]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / run, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"{name} = {run}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {s: jnp.clip(pre[s], 0.0, 1.0)
                     .reshape(-1, pre[s].shape[-1]).astype(jnp.float16)
                for s in sites}

    scatter = jax.jit(lambda a, c, t: a.at[t].add(c), donate_argnums=(0,))

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    acc = None
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        out = {s: np.asarray(v) for s, v in out.items()}  # plain host arrays
        if acc is None:
            acc = {s: jnp.zeros((VOCAB, v.shape[1]), jnp.float32)
                   for s, v in out.items()}
        tok = jnp.asarray(rows[i:i + B].reshape(-1))
        for s in sites:
            acc[s] = scatter(acc[s], jnp.asarray(out[s], jnp.float32), tok)
        if (i // B) % 25 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for s in sites:
        A = np.asarray(acc.pop(s))          # (vocab, C) summed CI
        top = np.argpartition(-A, K, axis=0)[:K].T      # (C, K), unsorted
        vals = np.take_along_axis(A.T, top, axis=1)
        order = np.argsort(-vals, axis=1)
        data[f"{s}|top_ids"] = np.take_along_axis(top, order, axis=1).astype(np.int32)
        data[f"{s}|top_ci"] = np.take_along_axis(vals, order, axis=1).astype(np.float32)
        data[f"{s}|total"] = A.sum(0, dtype=np.float64)
    np.savez_compressed(f"/data/top_tokens_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/top_tokens_{name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    list(compute.map(["newA", "newB"]))
