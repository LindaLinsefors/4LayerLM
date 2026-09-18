"""Modal job: reproduce the C/E decomposition runs' logged CEandKL eval metrics
under both RoPE spectra (configured base-1e4 vs fitted), to determine which
forward the DECOMPOSITION TRAINING itself ran with.

The WandB run p-d60af588 logged at step 100k (eval on pile_neox_tok_512_val):
  kl_ci_masked 0.618  kl_rounded_masked 0.592  kl_unmasked 0.0061
  kl_zero_masked 10.09  ce_difference_ci_masked -0.314
  ce_difference_rounded_masked -0.385  ce_difference_unmasked -0.025
Whichever spectrum reproduces those numbers (on our cached Pile rows -- different
eval data, so expect agreement to ~0.05, not exact) is the spectrum the trainer's
own eval -- and hence its training forward -- used. Mirrors
experiments/lm/eval.py::make_ce_kl_scorer (ci_masked / rounded@0.0 / unmasked /
zero_masked variants; stoch/random skipped -- they need RNG parity to compare).

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run --detach sink-models/hide/rope_ce_kl_modal.py
Output: cache/rope_ce_kl.npz + printed table.
"""

import modal

app = modal.App("rope-ce-kl-check")
vol = modal.Volume.from_name("vpd-4layer")

SINK_COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{SINK_COMMIT}"
    )
)

RUNS = {"C": "p-d60af588", "E": "p-bd411e35"}
N_ROWS = 128  # the runs' eval batch_size
B = 8


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600,
              memory=32768)
def ce_kl(run: str, log_inv_freq: list, tag: str) -> dict:
    from pathlib import Path

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import evaluate_ci
    from param_decomp.core.decomposed_linear import constrain_component_activation
    from param_decomp.core.model import MaterializedMasking, select_captures
    from param_decomp.core.precision import COMPUTE_DT
    from param_decomp.experiments.lm.eval import next_token_cross_entropy
    from param_decomp.experiments.lm.load_run import open_jax_run
    from param_decomp.targets import llama_simple_mlp as lsm
    from param_decomp.targets.losses import kl_per_position

    spec = jnp.asarray(np.exp(np.asarray(log_inv_freq, np.float64)), jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: spec

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / run, step=100000, data_root=root)
    placed, pw, ci_fn = loaded.placed, loaded.prepared_weights, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    rows = np.load("/data/pile_rows.npy")[:N_ROWS]

    @eqx.filter_jit
    def step(placed, pw, ci_fn, tokens):
        fwd = placed.clean_forward(tokens, keys)
        ci = evaluate_ci(ci_fn, select_captures(fwd.captures, keys),
                         remat=False).lower
        ci = {s: constrain_component_activation(v, placed.placement)
              for s, v in ci.items()}
        zeros_delta = {s: jnp.zeros_like(tokens, dtype=COMPUTE_DT)
                       for s in placed.site_names}
        variants = {
            "ci_masked": ci,
            "rounded_masked": {s: (v > 0.0).astype(COMPUTE_DT)
                               for s, v in ci.items()},
            "unmasked": {s: jnp.ones_like(v) for s, v in ci.items()},
            "zero_masked": {s: jnp.zeros_like(v) for s, v in ci.items()},
        }
        tgt_ce = next_token_cross_entropy(fwd.output, tokens)
        out = {"ce_target": tgt_ce}
        for name, masks in variants.items():
            m = placed.masked_forward(
                pw, tokens,
                masking=MaterializedMasking(component_masks=masks,
                                            weight_delta_masks=zeros_delta),
                capture_keys=frozenset(), remat=False).output
            out[f"kl_{name}"] = kl_per_position(m, fwd.output)
            out[f"ce_difference_{name}"] = (
                next_token_cross_entropy(m, tokens) - tgt_ce)
        return out

    acc = {}
    for i in range(0, N_ROWS, B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, pw, ci_fn, jnp.asarray(rows[i:i + B]))
        for k, v in out.items():
            acc.setdefault(k, []).append(float(v))
        print(f"{run} [{tag}] batch {i // B + 1}/{N_ROWS // B}", flush=True)
    return {k: float(np.mean(v)) for k, v in acc.items()}


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    lif_fitted = np.load(here / "cache" / "fitted_freqs_avg.npz")["log_inv_freq"]
    lif_cfg = -(np.arange(64) / 64) * np.log(1e4)

    handles = {}
    for name, run in RUNS.items():
        for tag, lif in [("fitted", lif_fitted), ("configured", lif_cfg)]:
            handles[(name, tag)] = ce_kl.spawn(run, lif.tolist(), tag)

    logged = {"kl_ci_masked": 0.618, "kl_rounded_masked": 0.592,
              "kl_unmasked": 0.0061, "kl_zero_masked": 10.09,
              "ce_difference_ci_masked": -0.314,
              "ce_difference_rounded_masked": -0.385,
              "ce_difference_unmasked": -0.025}
    results = {}
    for (name, tag), h in handles.items():
        results[f"{name}|{tag}"] = h.get()
    np.savez(here / "cache" / "rope_ce_kl.npz",
             **{f"{k}|{m}": v for k, r in results.items() for m, v in r.items()})
    for name in RUNS:
        print(f"\n== {name} ==   (WandB logged values for C shown right)")
        ks = sorted(results[f"{name}|fitted"])
        for k in ks:
            lg = f"  logged(C) {logged[k]:+.3f}" if k in logged else ""
            print(f"  {k:32s} configured {results[f'{name}|configured'][k]:+8.4f}"
                  f"   fitted {results[f'{name}|fitted'][k]:+8.4f}{lg}")
