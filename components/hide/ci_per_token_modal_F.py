"""Modal GPU job: per-token-id CI sums for the top-20 sample-mean-CI components
of h.0.attn.q_proj AND h.0.attn.k_proj of decomposition F (p-c45e0001, the
corrected-RoPE re-decomposition of C's target t-87f91319) over the 4,000
cached Pile rows — the F analogue of ci_per_token_modal_{C,DE}.py.

RoPE: F was TRAINED with the fitted corrected spectrum
(sink-models/hide/cache/fitted_freqs_avg.npz), so the patch here is F's actual
training-time forward (no broken-RoPE caveat, unlike C/D/E). NLL assert in-job.

Output per matrix: cache/ci_per_token_F_<h0q|h0k>_top20.npz
("ids", "ci_sum", "fires") — summed CI and fire counts (CI > 0.1) per input
token id, (vocab, 20); ids selected locally from coci-heatmaps coci_F.npz
(fetch it first: modal volume get vpd-4layer /coci_F.npz coci-heatmaps/hide/cache/).

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run components/hide/ci_per_token_modal_F.py
"""

import modal

app = modal.App("components-ci-per-token-f")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUN = "p-c45e0001"  # F
STEP = 100000
VOCAB = 50277
N_TOP = 20
FIRE_THRESH = 0.1
MODS = ("h.0.attn.q_proj", "h.0.attn.k_proj")


def short_name(mod: str) -> str:  # same rule as components/C
    p = mod.split(".")
    return f"h{p[1]}{p[3][0] if p[2] == 'attn' else p[3][0] + 'm'}"


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(comps_by_mod: dict, inv_freq: list) -> dict:
    import time
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")

    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.targets import llama_simple_mlp as lsm
    from param_decomp.experiments.lm.load_run import open_jax_run

    fitted = jnp.asarray(np.asarray(inv_freq, np.float64), jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: fitted

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / RUN, step=STEP, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    idx = {m: jnp.asarray(np.asarray(c, np.int32)) for m, c in comps_by_mod.items()}

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32

    # sanity: with the fitted spectrum the target must be good (~2.5 NLL)
    with jax.set_mesh(loaded.mesh):
        logits = placed.clean_forward(jnp.asarray(rows[:8]), keys).output
    logits = np.asarray(logits, np.float32)
    lse = np.log(np.exp(logits[:, :-1] - logits[:, :-1].max(-1, keepdims=True)
                        ).sum(-1)) + logits[:, :-1].max(-1)
    tgt = rows[:8, 1:]
    nll = float((lse - np.take_along_axis(
        logits[:, :-1], tgt[..., None], -1)[..., 0]).mean())
    print(f"target NLL (first 8 rows, fitted RoPE): {nll:.3f}", flush=True)
    assert nll < 4.0, f"RoPE patch did not take effect (NLL {nll:.2f})"

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {m: jnp.clip(pre[m], 0.0, 1.0)
                .reshape(-1, pre[m].shape[-1])[:, idx[m]] for m in comps_by_mod}

    scatter = jax.jit(
        lambda a, f, c, t: (a.at[t].add(c), f.at[t].add(c > FIRE_THRESH)),
        donate_argnums=(0, 1))

    B = 16
    acc = {m: jnp.zeros((VOCAB, len(c)), jnp.float32) for m, c in comps_by_mod.items()}
    fires = {m: jnp.zeros((VOCAB, len(c)), jnp.int32) for m, c in comps_by_mod.items()}
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        tok = jnp.asarray(rows[i:i + B].reshape(-1))
        for m in comps_by_mod:
            c = jnp.asarray(np.asarray(out[m], np.float32))   # off-mesh copy
            acc[m], fires[m] = scatter(acc[m], fires[m], c, tok)
        if (i // B) % 50 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"done, {time.time() - t0:.0f}s", flush=True)
    return {m: {"ci_sum": np.asarray(acc[m]), "fires": np.asarray(fires[m])}
            for m in comps_by_mod}


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    coci = np.load(here.parents[1] / "coci-heatmaps" / "hide" / "cache"
                   / "coci_F.npz")
    comps = {m: np.argsort(coci[f"{m}|mean"])[::-1][:N_TOP].tolist()
             for m in MODS}
    for m in MODS:
        print(f"F top {N_TOP} of {m}: {comps[m]}")
    lif = np.exp(np.load(here.parents[1] / "sink-models" / "hide" / "cache"
                         / "fitted_freqs_avg.npz")["log_inv_freq"]).tolist()

    (here / "cache").mkdir(exist_ok=True)
    out = compute.remote(comps, lif)
    for m in MODS:
        path = here / "cache" / f"ci_per_token_F_{short_name(m)}_top20.npz"
        np.savez_compressed(path, ids=np.asarray(comps[m], np.int32),
                            ci_sum=out[m]["ci_sum"], fires=out[m]["fires"])
        print("saved", path)
