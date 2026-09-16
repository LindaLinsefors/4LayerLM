"""Modal GPU job: per-token-id CI statistics for decomposition C's pos-0
components (the >90%-of-fires-at-position-0 set from pos0_analysis).

The member list ({site: [comp ids]}) is computed locally from
sink-models/hide/cache/pos_fires_sink_C.npz (alive: sample mean CI > 1e-6;
total fires >= 20; pos-0 fire share > SHARE) and passed as an argument; the job accumulates, for just those K
components, summed CI and fire counts (CI > 0.1) per *input token id* over the
4,000 cached Pile rows, and returns the full (vocab, K) arrays over the wire
(~25 MB) — saved locally to cache/top_tokens_Cpos0.npz: "site" (K, index into
the returned site order), "sites" (site names), "ids" (K,), "ci" (vocab, K)
float32, "fires" (vocab, K) int32.

Requires the sink bundles on volume vpd-4layer (put there by
sink-models/hide/pos_fires_sink_modal.py's fetch) and /pile_rows.npy.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run C-pos-0/hide/top_tokens_modal.py
"""

import modal

app = modal.App("c-pos0-top-tokens")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUN = "p-d60af588"  # C
STEP = 100000
VOCAB = 50277
FIRE_THRESH = 0.1
SHARE = 0.3


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(members: dict) -> dict:
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

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / RUN, step=STEP, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    msites = [s for s in ci_fn.fn.output_names if s in members]
    idx = {s: jnp.asarray(np.asarray(members[s], np.int32)) for s in msites}
    K = sum(len(members[s]) for s in msites)
    print(f"{RUN}: {K} pos-0 components over {len(msites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        cols = [jnp.clip(pre[s], 0.0, 1.0).reshape(-1, pre[s].shape[-1])[:, idx[s]]
                for s in msites]
        return jnp.concatenate(cols, axis=1)                  # (t, K)

    scatter = jax.jit(
        lambda a, f, c, t: (a.at[t].add(c), f.at[t].add(c > FIRE_THRESH)),
        donate_argnums=(0, 1))

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    acc = jnp.zeros((VOCAB, K), jnp.float32)
    fires = jnp.zeros((VOCAB, K), jnp.int32)
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            c = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        c = jnp.asarray(np.asarray(c, np.float32))            # off-mesh copy
        tok = jnp.asarray(rows[i:i + B].reshape(-1))
        acc, fires = scatter(acc, fires, c, tok)
        if (i // B) % 50 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"done, {time.time() - t0:.0f}s", flush=True)
    out = {"order": msites, "ci": np.asarray(acc), "fires": np.asarray(fires)}
    # also persist on the volume: the detached local client may be gone by now
    np.savez_compressed("/data/top_tokens_Cpos0_raw.npz",
                        order=np.array(msites), ci=out["ci"], fires=out["fires"])
    vol.commit()
    return out


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    z = np.load(here.parents[1] / "sink-models" / "hide" / "cache"
                / "pos_fires_sink_C.npz")
    mods = sorted({k.split("|")[0] for k in z.files})
    members = {}
    for m in mods:
        fp = z[f"{m}|Fp"]
        tot = fp.sum(1)
        share = np.divide(fp[:, 0], tot, out=np.zeros(len(fp)), where=tot > 0)
        alive = z[f"{m}|Sp"].sum(1) / (4000 * 512) > 1e-6
        ids = np.nonzero(alive & (tot >= 20) & (share > SHARE))[0]
        if len(ids):
            members[m] = ids.tolist()
    K = sum(len(v) for v in members.values())
    print(f"{K} selected components over {len(members)} matrices")

    out = compute.remote(members)
    site = np.concatenate([np.full(len(members[s]), out["order"].index(s), np.int16)
                           for s in out["order"]])
    ids = np.concatenate([np.asarray(members[s], np.int32) for s in out["order"]])
    (here / "cache").mkdir(exist_ok=True)
    np.savez_compressed(here / "cache" / "top_tokens_Cpos0.npz",
                        sites=np.array(out["order"]), site=site, ids=ids,
                        ci=out["ci"], fires=out["fires"])
    print("saved", here / "cache" / "top_tokens_Cpos0.npz")
