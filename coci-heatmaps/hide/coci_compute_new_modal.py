"""Modal GPU job: per-matrix co-CI correlation matrices for the two NEW 800k-step
JAX decompositions of the pile_4l target (see CLAUDE.md "New 800k-step
decompositions"):

  newA = p-8383f5e5  (ImportanceMinimalityLoss frequency coeff 6.6e-5)
  newB = p-4d9a6a12  (coeff 6.6e-6)

Same statistics as coci_compute_modal.py, computed with the JAX library at the
pinned public commit instead of the torch ComponentModel: per site (24 matrices,
h.<l>.{attn.{q,k,v,o}_proj, mlp.{c_fc,down_proj}}, C=768 attn / 3072 mlp),
per-token CI = clip(preactivations, 0, 1) (identical to torch lower_leaky with
sampling="continuous"), over the 4,000 cached Pile rows (2.05M tokens).
Batch sums/Grams are computed on-GPU in fp32 (matmul precision "highest"),
accumulated on host in float64; r saved as float16.

The archives are fetched straight from the public WandB share links into the
volume (no local upload); /pile_rows.npy is the int32 (4000, 512) conversion of
context-loss's pile_rows.pt (the JAX image has no torch).

Run:  modal run coci-heatmaps/hide/coci_compute_new_modal.py
Then: modal volume get vpd-4layer /coci_newA.npz coci-heatmaps/hide/cache/
      modal volume get vpd-4layer /coci_newB.npz coci-heatmaps/hide/cache/
Output npz keys per module: "<mod>|mean" (C, float64), "<mod>|F" (C, float64),
"<mod>|r" (C x C, float16); plus "T" (token count).
"""

import modal

app = modal.App("coci-new-decomps")
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
BASE = "https://api.wandb.ai/files/goodfire/param-decomp/share-task-1373"


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch() -> None:
    """Download + extract both decomposition archives into the volume (idempotent)."""
    import subprocess
    from pathlib import Path

    dest = Path("/data/new-decomps")
    dest.mkdir(exist_ok=True)
    for run in RUNS.values():
        name = f"{run}-decomposition-800000"
        if (dest / name / "runs" / run / "ckpts").exists():
            print(f"{name} already present", flush=True)
            continue
        print(f"fetching {name} ...", flush=True)
        subprocess.run(
            f"curl -sL '{BASE}/{name}.tar.zst' | zstd -d | "
            f"tar -x --no-same-owner -C {dest}",
            shell=True, check=True)
        print(f"extracted {name}", flush=True)
    # orbax stats every checkpoint file and resolves its owner via getpwuid —
    # uids preserved from the archive don't exist in the container
    subprocess.run(f"chown -R 0:0 {dest}", shell=True, check=True)
    vol.commit()


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

    FIRE_THRESH = 0.1
    run = RUNS[name]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / run, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"{name} = {run}: {len(sites)} sites", flush=True)

    from jax.sharding import PartitionSpec as P

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        out = {}
        for s in sites:
            c = jnp.clip(pre[s], 0.0, 1.0).reshape(-1, pre[s].shape[-1])
            # token dim is mesh-sharded; the Gram contracts over it, so the
            # (replicated) output sharding must be stated explicitly
            g = jnp.einsum("tc,td->cd", c, c, out_sharding=P(None, None))
            out[s] = (c.sum(0), (c > FIRE_THRESH).sum(0), g)
        return out

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    S1 = F = G = None
    T = 0
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            tok = jnp.asarray(rows[i:i + B])
            out = step(placed, ci_fn, tok)
            if S1 is None:
                S1 = {s: np.zeros(out[s][0].shape[0], np.float64) for s in sites}
                F = {s: np.zeros_like(S1[s]) for s in sites}
                G = {s: np.zeros((len(S1[s]), len(S1[s])), np.float64) for s in sites}
            for s in sites:
                S1[s] += np.asarray(out[s][0], np.float64)
                F[s] += np.asarray(out[s][1], np.float64)
                G[s] += np.asarray(out[s][2], np.float64)
            T += tok.size
            if (i // B) % 25 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, {T} tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)

    data = {"T": np.array(T)}
    for s in sites:
        mean = S1[s] / T
        cov = G[s] / T - np.outer(mean, mean)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov / np.outer(sd, sd)
        r[~np.isfinite(r)] = np.nan
        data[f"{s}|mean"] = mean
        data[f"{s}|F"] = F[s]
        data[f"{s}|r"] = r.astype(np.float16)
    np.savez(f"/data/coci_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/coci_{name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    fetch.remote()
    list(compute.map(["newA", "newB"]))
