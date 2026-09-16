"""Modal GPU job: per-position CI fire counts for the two NEW 800k-step
decompositions (newA = p-8383f5e5, newB = p-4d9a6a12).

For every component of every matrix: how often it fires (CI > 0.1, the
project's usual threshold) at each chunk position, over the 4,000 cached Pile
rows. The newA/newB analogue of pile-qk-comps' pos_ci.npz fire counts —
used e.g. to flag pos-0 (attention-sink) components in hovers.

Same forward/CI pipeline as top_tokens_compute_modal.py (mean-ci-widget);
accumulation is on host (a (512, C) int per site is tiny).

Run:  modal run coci-heatmaps/hide/pos_fires_new_modal.py
Then: modal volume get vpd-4layer /pos_fires_newA.npz coci-heatmaps/hide/cache/
      modal volume get vpd-4layer /pos_fires_newB.npz coci-heatmaps/hide/cache/
Output npz keys per module: "<mod>|Fp" (C, 512) int32 fire counts per position.
"""

import modal

app = modal.App("pos-fires-new-decomps")
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
FIRE_THRESH = 0.1


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
        return {s: jnp.clip(pre[s], 0.0, 1.0).astype(jnp.float16)
                for s in sites}  # (B, T, C)

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    acc = None
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        if acc is None:
            acc = {s: np.zeros((rows.shape[1], v.shape[-1]), np.int64)
                   for s, v in out.items()}
        for s in sites:
            acc[s] += (np.asarray(out[s]) > FIRE_THRESH).sum(0)
        if (i // B) % 25 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    data = {f"{s}|Fp": acc[s].T.astype(np.int32) for s in sites}  # (C, 512)
    np.savez_compressed(f"/data/pos_fires_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/pos_fires_{name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    list(compute.map(["newA", "newB"]))
