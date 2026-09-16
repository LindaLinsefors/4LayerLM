"""Modal GPU job: where/on what does newB's giant h.3.attn.k_proj co-CI cluster
fire? (85 members, from report_alive_clustered09.md; ids re-derived locally by
rule_clusters and hardcoded below.)

For the 85 components: per-component per-position CI>0.1 fire counts over the
4,000 cached Pile rows, plus the full (row, pos, comp) table of all fires at
pos > 0 (sparse: ~30k entries) so tokens/contexts can be decoded locally from
pile_rows.pt.

Run:  modal run coci-heatmaps/hide/newB_bigcluster_modal.py
Then: modal volume get vpd-4layer /bigcluster_newB.npz coci-heatmaps/hide/cache/
Output npz: ids (85,), pos_counts (85, 512), fires_row/pos/comp (N,) int32.
"""

import modal

app = modal.App("newB-bigcluster")
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
SITE = "h.3.attn.k_proj"
IDS = [50, 57, 61, 64, 69, 109, 114, 117, 118, 120, 124, 160, 173, 178, 180,
       183, 185, 186, 195, 209, 225, 228, 231, 239, 240, 243, 245, 254, 297,
       303, 308, 311, 317, 356, 361, 364, 369, 370, 374, 375, 377, 379, 381,
       436, 441, 446, 492, 497, 498, 506, 508, 509, 511, 539, 549, 550, 561,
       564, 569, 573, 590, 621, 623, 625, 635, 636, 681, 682, 689, 693, 696,
       697, 699, 700, 701, 703, 751, 753, 754, 755, 756, 759, 761, 765, 766]
FIRE_THRESH = 0.1


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute() -> None:
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

    root = Path(f"/data/new-decomps/{RUN}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / RUN, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    ids = jnp.asarray(np.array(IDS))

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        ci = jnp.clip(pre[SITE][..., ids], 0.0, 1.0)  # (B, S, 85)
        return ci > FIRE_THRESH

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    pos_counts = np.zeros((len(IDS), rows.shape[1]), np.int64)
    fr, fp, fc = [], [], []
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            fires = np.asarray(step(placed, ci_fn, jnp.asarray(rows[i:i + B])))
            pos_counts += fires.sum(0).T
            b, p, c = np.nonzero(fires[:, 1:, :])  # pos > 0 only
            fr.append(b.astype(np.int32) + i)
            fp.append(p.astype(np.int32) + 1)
            fc.append(c.astype(np.int32))
            if (i // B) % 50 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    np.savez("/data/bigcluster_newB.npz", ids=np.array(IDS),
             pos_counts=pos_counts, fires_row=np.concatenate(fr),
             fires_pos=np.concatenate(fp), fires_comp=np.concatenate(fc))
    vol.commit()
    print(f"saved /data/bigcluster_newB.npz, {sum(len(x) for x in fp)} "
          f"pos>0 fires ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    compute.remote()
