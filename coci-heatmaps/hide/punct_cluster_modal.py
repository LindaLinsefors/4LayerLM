"""Modal GPU job: all fire sites (row, pos, comp) of newB's 18-comp
h.2.mlp.c_fc punctuation cluster (the one non-pos-0 large newB cluster from
big_clusters_modal.py), so contexts can be decoded locally.

Run:  modal run coci-heatmaps/hide/punct_cluster_modal.py
Then: modal volume get vpd-4layer /punct_cluster_newB.npz coci-heatmaps/hide/cache/
"""

from pathlib import Path

import modal

app = modal.App("newB-punct-cluster")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUN = "p-4d9a6a12"  # newB
SITE = "h.2.mlp.c_fc"
FIRE_THRESH = 0.1


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(ids_in: list[int]) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")

    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run

    root = Path(f"/data/new-decomps/{RUN}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / RUN, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    ids = jnp.asarray(np.array(ids_in))

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return jnp.clip(pre[SITE][..., ids], 0.0, 1.0) > FIRE_THRESH

    rows = np.load("/data/pile_rows.npy")
    B = 16
    fr, fp, fc = [], [], []
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            fires = np.asarray(step(placed, ci_fn, jnp.asarray(rows[i:i + B])))
            b, p, c = np.nonzero(fires)
            fr.append(b.astype(np.int32) + i)
            fp.append(p.astype(np.int32))
            fc.append(c.astype(np.int32))

    np.savez("/data/punct_cluster_newB.npz", ids=np.array(ids_in),
             fires_row=np.concatenate(fr), fires_pos=np.concatenate(fp),
             fires_comp=np.concatenate(fc))
    vol.commit()
    print(f"saved, {sum(len(x) for x in fp)} fires ({time.time() - t0:.0f}s)")


@app.local_entrypoint()
def main() -> None:
    import json
    import os

    clusters = json.loads(Path(os.environ["BIG_CLUSTERS_JSON"]).read_text())
    (ids,) = [c["ids"] for c in clusters["newB"]
              if c["site"] == SITE and len(c["ids"]) == 18]
    compute.remote(ids)
