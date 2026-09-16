"""Modal GPU job: co-CI statistics + top activating tokens for the three
attention-sink decompositions (see CLAUDE.md "Attention-sink models &
decompositions" and models_and_decomps.md):

  C = p-d60af588  (target t-87f91319, decomposition seed 0)
  D = p-fecd6a6b  (target t-87f91319, decomposition seed 1)
  E = p-bd411e35  (target t-75f6c439, decomposition seed 0)

One pass over the 4,000 cached Pile rows (2.05M tokens) per run computes both
outputs at once:
  /data/coci_{C,D,E}.npz        — same format as coci_{newA,newB}.npz:
                                  "<mod>|mean", "<mod>|F" (C, float64),
                                  "<mod>|r" (C x C, float16), "T"
  /data/top_tokens_{C,D,E}.npz  — same format as top_tokens_{newA,newB}.npz:
                                  "<mod>|top_ids"/"|top_ci" (C, K), "<mod>|total"

CI = clip(preactivations, 0, 1) as in coci_compute_new_modal.py. The library
must be the sink-loader PR #1002 commit (82a67f71c) — the pinned public main
cannot load untied-head/attention-sink targets. The five task-1423 archives
(3 decompositions + 2 bases) extract into one shared root /data/sink-models/
(runs/ + pretrain_cache/), unlike the per-archive roots of new-decomps.

Run:  modal run coci-heatmaps/hide/coci_compute_sink_modal.py
Then: modal volume get vpd-4layer /coci_C.npz coci-heatmaps/hide/cache/  (etc.)
      modal volume get vpd-4layer /top_tokens_C.npz mean-ci-widget/hide/cache/
"""

import modal

app = modal.App("coci-sink-decomps")
vol = modal.Volume.from_name("vpd-4layer")

# PR #1002 branch bridge/task-1423-external-loader (sink-aware loader)
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
VOCAB = 50277
K = 20


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch() -> None:
    """Download + extract all five task-1423 archives into /data/sink-models
    (idempotent). All archives share one data root."""
    import subprocess
    from pathlib import Path

    vol.reload()  # a previous run's commit may still have been propagating
    dest = Path("/data/sink-models")
    dest.mkdir(exist_ok=True)
    # per archive, ALL files the consumers need (a partially-visible volume
    # commit once left ckpts/ present but deliverable.yaml missing)
    done = {
        f"{run}-decomposition-step{STEP}": [
            dest / "runs" / run / "deliverable.yaml",
            dest / "runs" / run / "launch_config.yaml",
            dest / "runs" / run / "ckpts" / str(STEP) / "decomposition",
        ]
        for run in RUNS.values()
    } | {
        f"{t}-base-step{STEP}": [
            dest / "pretrain_cache" / f"spd-{t}" / "model_config.yaml",
            dest / "pretrain_cache" / f"spd-{t}" / f"model_step_{STEP}.safetensors",
        ]
        for t in BASES
    }
    for name, required in done.items():
        if all(p.exists() for p in required):
            print(f"{name} already present", flush=True)
            continue
        print(f"fetching {name} ...", flush=True)
        subprocess.run(
            f"curl -sL '{BASE_URL}/{name}.tar.zst' | zstd -d | "
            f"tar -x --no-same-owner -C {dest}",
            shell=True, check=True)
        print(f"extracted {name}", flush=True)
    # orbax stats every checkpoint file and resolves its owner via getpwuid —
    # uids preserved from the archive don't exist in the container
    subprocess.run(f"chown -R 0:0 {dest}", shell=True, check=True)

    # the deliverables were written by a newer internal revision whose
    # LMTargetConfig has an `output_edge` field; the PR-commit schema forbids
    # extras, and `materialized` is the only behavior it implements anyway —
    # strip the key from the VOLUME copies (local sink-models/ stays pristine)
    import yaml
    for run in RUNS.values():
        p = dest / "runs" / run / "deliverable.yaml"
        doc = yaml.safe_load(p.read_text())
        if doc.get("target", {}).pop("output_edge", None) is not None:
            p.write_text(yaml.safe_dump(doc, sort_keys=False))
            print(f"stripped target.output_edge from {p}", flush=True)
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

    from jax.sharding import PartitionSpec as P
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run

    FIRE_THRESH = 0.1
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
        out = {}
        for s in sites:
            c = jnp.clip(pre[s], 0.0, 1.0).reshape(-1, pre[s].shape[-1])
            # token dim is mesh-sharded; the Gram contracts over it, so the
            # (replicated) output sharding must be stated explicitly
            g = jnp.einsum("tc,td->cd", c, c, out_sharding=P(None, None))
            out[s] = (c.sum(0), (c > FIRE_THRESH).sum(0), g,
                      c.astype(jnp.float16))
        return out

    scatter = jax.jit(lambda a, c, t: a.at[t].add(c), donate_argnums=(0,))

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    S1 = F = G = acc = None
    T = 0
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        out = {s: (np.asarray(v[0]), np.asarray(v[1]), np.asarray(v[2]),
                   np.asarray(v[3])) for s, v in out.items()}  # host arrays
        if S1 is None:
            S1 = {s: np.zeros(v[0].shape[0], np.float64)
                  for s, v in out.items()}
            F = {s: np.zeros_like(S1[s]) for s in sites}
            G = {s: np.zeros((len(S1[s]), len(S1[s])), np.float64)
                 for s in sites}
            acc = {s: jnp.zeros((VOCAB, len(S1[s])), jnp.float32)
                   for s in sites}
        tok = jnp.asarray(rows[i:i + B].reshape(-1))
        for s in sites:
            S1[s] += out[s][0].astype(np.float64)
            F[s] += out[s][1].astype(np.float64)
            G[s] += out[s][2].astype(np.float64)
            acc[s] = scatter(acc[s], jnp.asarray(out[s][3], jnp.float32), tok)
        T += rows[i:i + B].size
        if (i // B) % 25 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, {T} tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)

    coci = {"T": np.array(T)}
    top = {}
    for s in sites:
        mean = S1[s] / T
        cov = G[s] / T - np.outer(mean, mean)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov / np.outer(sd, sd)
        r[~np.isfinite(r)] = np.nan
        coci[f"{s}|mean"] = mean
        coci[f"{s}|F"] = F[s]
        coci[f"{s}|r"] = r.astype(np.float16)

        A = np.asarray(acc.pop(s))          # (vocab, C) summed CI
        ids = np.argpartition(-A, K, axis=0)[:K].T      # (C, K), unsorted
        vals = np.take_along_axis(A.T, ids, axis=1)
        order = np.argsort(-vals, axis=1)
        top[f"{s}|top_ids"] = np.take_along_axis(ids, order, axis=1).astype(np.int32)
        top[f"{s}|top_ci"] = np.take_along_axis(vals, order, axis=1).astype(np.float32)
        top[f"{s}|total"] = A.sum(0, dtype=np.float64)
    np.savez(f"/data/coci_{name}.npz", **coci)
    np.savez_compressed(f"/data/top_tokens_{name}.npz", **top)
    vol.commit()
    print(f"saved /data/coci_{name}.npz + /data/top_tokens_{name}.npz "
          f"({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    fetch.remote()
    list(compute.map(list(RUNS)))
