"""Modal GPU job: per-token-id CI sums for the top-20 mean-CI components of one
matrix of decomposition C over the 4,000 cached Pile rows (adapted from
C-pos-0/hide/top_tokens_modal.py; ids selected locally from coci_C.npz).

Accumulates summed CI and fire counts (CI > 0.1) per input token id, returns the
(vocab, 20) arrays over the wire -> cache/ci_per_token_<short>_top20.npz
("ids", "ci_sum", "fires"), short = e.g. h0q for h.0.attn.q_proj. Mean CI per
token id = ci_sum / corpus count (counts computed locally).

⚠ Uses the public JAX sink loader = broken RoPE (see sink-models/rope_report.md);
token-identity-level CI statistics are expected to survive qualitatively.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run components/hide/ci_per_token_modal_C.py [--mod h.0.attn.k_proj]
"""

import modal

app = modal.App("c-components-ci-per-token")
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
N_TOP = 20
FIRE_THRESH = 0.1


def short_name(mod: str) -> str:
    p = mod.split(".")  # h.<l>.attn.<m>_proj / h.<l>.mlp.<m>
    return f"h{p[1]}{p[3][0] if p[2] == 'attn' else p[3][0] + 'm'}"


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(comps: list, mod: str) -> dict:
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
    idx = jnp.asarray(np.asarray(comps, np.int32))
    K = len(comps)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return jnp.clip(pre[mod], 0.0, 1.0).reshape(-1, pre[mod].shape[-1])[:, idx]

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
    return {"ci_sum": np.asarray(acc), "fires": np.asarray(fires)}


@app.local_entrypoint()
def main(mod: str = "h.0.attn.q_proj") -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    mean = np.load(here.parents[1] / "compare-decomps" / "hide" / "cache"
                   / "coci_C.npz")[f"{mod}|mean"]
    comps = np.argsort(mean)[::-1][:N_TOP].tolist()
    print(f"top {N_TOP} of {mod} by sample mean CI: {comps}")

    out = compute.remote(comps, mod)
    (here / "cache").mkdir(exist_ok=True)
    path = here / "cache" / f"ci_per_token_{short_name(mod)}_top20.npz"
    np.savez_compressed(path, ids=np.asarray(comps, np.int32),
                        ci_sum=out["ci_sum"], fires=out["fires"])
    print("saved", path)
