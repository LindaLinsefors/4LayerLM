"""Modal GPU job: per-position CI statistics for the three attention-sink
decompositions (see CLAUDE.md "Attention-sink models & decompositions"):

  C = p-d60af588  (target t-87f91319, seed 0)
  D = p-fecd6a6b  (target t-87f91319, seed 1)
  E = p-bd411e35  (target t-75f6c439, seed 0)

Question: do pos-0 (chunk-start attention-sink) components still exist when the
base model has a *built-in* learned sink slot? For every component of every
matrix, over the 4,000 cached Pile rows: fire counts (CI > 0.1) and CI sums per
chunk position — the sink-model analogue of pos_fires_{newA,newB}.npz.

Loader: the PR #1002 branch commit 82a67f71c (the pinned public commit cannot
load untied-head/sink models). NOTE: the bundle README's
`layout=SINGLE_DEVICE_RESIDENT_LAYOUT` does not exist at this commit —
`open_jax_run(run_dir, step, data_root=...)` builds its own single-device mesh.
Archives are fetched straight from the public share links into the volume.

Run:  modal run sink-models/hide/pos_fires_sink_modal.py
Then: modal volume get vpd-4layer /pos_fires_sink_C.npz sink-models/hide/cache/
      (same for _D, _E)
Output npz keys per module: "<mod>|Fp" (C, 512) int32 fire counts per position,
"<mod>|Sp" (C, 512) float32 CI sums per position.
"""

import modal

app = modal.App("pos-fires-sink-decomps")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35"}
BASES = ["t-87f91319", "t-75f6c439"]
BASE_URL = "https://api.wandb.ai/files/goodfire/param-decomp/share-task-1423"
STEP = 100000
FIRE_THRESH = 0.1


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch() -> None:
    """Download + extract all five archives into one data root; a `.ok_<name>`
    marker guards against partially extracted archives (an interrupted first
    attempt left one). Also strips `target.output_edge` from each run's
    deliverable.yaml: the shipped deliverables were authored by a newer internal
    schema, and LMTargetConfig at the public PR commit forbids the extra field —
    it is redundant (untied head + sinks are declared in the base's
    model_config.yaml, which the target builder reads)."""
    import shutil
    import subprocess
    from pathlib import Path

    import yaml

    dest = Path("/data/sink-models")
    dest.mkdir(exist_ok=True)
    archives = [(f"{r}-decomposition-step{STEP}", Path("runs") / r)
                for r in RUNS.values()]
    archives += [(f"{t}-base-step{STEP}", Path("pretrain_cache") / f"spd-{t}")
                 for t in BASES]
    for name, out_dir in archives:
        marker = dest / f".ok_{name}"
        if marker.exists():
            print(f"{name} already present", flush=True)
            continue
        shutil.rmtree(dest / out_dir, ignore_errors=True)
        print(f"fetching {name} ...", flush=True)
        subprocess.run(
            f"curl -sL '{BASE_URL}/{name}.tar.zst' | zstd -d | "
            f"tar -x --no-same-owner -C {dest}",
            shell=True, check=True)
        marker.touch()
        print(f"extracted {name}", flush=True)
    # orbax stats every checkpoint file and resolves its owner via getpwuid —
    # uids preserved from the archive don't exist in the container
    subprocess.run(f"chown -R 0:0 {dest}", shell=True, check=True)
    for run in RUNS.values():
        path = dest / "runs" / run / "deliverable.yaml"
        cfg = yaml.safe_load(path.read_text())
        if cfg["target"].pop("output_edge", None) is not None:
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f"stripped output_edge from {run}/deliverable.yaml", flush=True)
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

    run = RUNS[name]
    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / run, step=STEP, data_root=root)
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
    Fp = Sp = None
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        if Fp is None:
            Fp = {s: np.zeros((rows.shape[1], v.shape[-1]), np.int64)
                  for s, v in out.items()}
            Sp = {s: np.zeros_like(Fp[s], np.float64) for s in sites}
        for s in sites:
            ci = np.asarray(out[s], np.float32)
            Fp[s] += (ci > FIRE_THRESH).sum(0)
            Sp[s] += ci.sum(0)
        if (i // B) % 25 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for s in sites:
        data[f"{s}|Fp"] = Fp[s].T.astype(np.int32)   # (C, 512)
        data[f"{s}|Sp"] = Sp[s].T.astype(np.float32)
    np.savez_compressed(f"/data/pos_fires_sink_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/pos_fires_sink_{name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    fetch.remote()
    list(compute.map(list(RUNS)))
