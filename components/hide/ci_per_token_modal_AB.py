"""Modal GPU job: per-token-id CI sums for the top-20 sample-mean-CI components
of h.0.attn.q_proj AND h.0.attn.k_proj of the newA/newB decompositions
(p-8383f5e5 / p-4d9a6a12, pile_4l target t-9d2b8f02, JAX 800k-step checkpoints)
over the 4,000 cached Pile rows. Loading per coci-heatmaps'
coci_compute_new_modal.py (pinned public commit, archives already on the
volume under /new-decomps/); accumulation per ci_per_token_modal_C.py.
No RoPE caveat: the pile_4l target loads correctly at the pinned commit.

Output per decomposition x matrix: cache/ci_per_token_{A,B}_<h0q|h0k>_top20.npz
("ids", "ci_sum", "fires") — summed CI and fire counts (CI > 0.1) per input
token id, (vocab, 20); component ids selected locally from coci_{newA,newB}.npz.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run components/hide/ci_per_token_modal_AB.py
"""

import modal

app = modal.App("components-ci-per-token-ab")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"A": "p-8383f5e5", "B": "p-4d9a6a12"}
STEP = 800000
VOCAB = 50277
N_TOP = 20
FIRE_THRESH = 0.1
MODS = ("h.0.attn.q_proj", "h.0.attn.k_proj")


def short_name(mod: str) -> str:  # same rule as components/C
    p = mod.split(".")
    return f"h{p[1]}{p[3][0] if p[2] == 'attn' else p[3][0] + 'm'}"


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(dec: str, comps_by_mod: dict) -> dict:
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

    run = RUNS[dec]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    assert (root / "runs" / run / "ckpts").exists(), f"archive missing: {root}"
    loaded = open_jax_run(root / "runs" / run, step=STEP, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    idx = {m: jnp.asarray(np.asarray(c, np.int32)) for m, c in comps_by_mod.items()}

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {m: jnp.clip(pre[m], 0.0, 1.0)
                .reshape(-1, pre[m].shape[-1])[:, idx[m]] for m in comps_by_mod}

    scatter = jax.jit(
        lambda a, f, c, t: (a.at[t].add(c), f.at[t].add(c > FIRE_THRESH)),
        donate_argnums=(0, 1))

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
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
            print(f"{dec} batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"{dec} done, {time.time() - t0:.0f}s", flush=True)
    return {m: {"ci_sum": np.asarray(acc[m]), "fires": np.asarray(fires[m])}
            for m in comps_by_mod}


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    jobs = []
    for dec, coci_name in (("A", "newA"), ("B", "newB")):
        coci = np.load(here.parents[1] / "coci-heatmaps" / "hide" / "cache"
                       / f"coci_{coci_name}.npz")
        comps = {m: np.argsort(coci[f"{m}|mean"])[::-1][:N_TOP].tolist()
                 for m in MODS}
        for m in MODS:
            print(f"{dec} top {N_TOP} of {m}: {comps[m]}")
        jobs.append((dec, comps))

    (here / "cache").mkdir(exist_ok=True)
    for (dec, comps), out in zip(jobs, compute.starmap(jobs)):
        for m in MODS:
            path = here / "cache" / f"ci_per_token_{dec}_{short_name(m)}_top20.npz"
            np.savez_compressed(path, ids=np.asarray(comps[m], np.int32),
                                ci_sum=out[m]["ci_sum"], fires=out[m]["fires"])
            print("saved", path)
