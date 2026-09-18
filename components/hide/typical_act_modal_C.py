"""Modal GPU job: typical activation strength per component, decomposition C.

For each component c of the requested matrices, accumulates over the 4,000
cached Pile rows (2.05M tokens; empirical corpus frequency = the token-frequency
weighting):

    Sci   = sum_pos CI_c(pos)                 (continuous CI = clip(preact,0,1))
    Sa_ci = sum_pos CI_c(pos) * a_c(pos)      (a = input activation x @ V_c)
    F     = sum_pos [CI_c(pos) > 0.1]
    Sa_f  = sum_pos a_c(pos) * [CI_c(pos) > 0.1]

Typical activation = Sa_ci / Sci (CI-weighted mean of a, token-frequency
weighted through the sample); Sa_f / F is the CI>0.1-indicator variant kept for
comparison. Saved with the raw sums to cache/typical_act_C.npz (keys
"<mod>|Sci" etc. + "<mod>|typical") so other weightings can be recomputed.

RoPE: the loader's plain_rope_inv_freq is monkeypatched with the FITTED
corrected spectrum (sink-models/hide/cache/fitted_freqs_avg.npz, passed over
the wire) -- the standard for all C/D/E forwards until the co-authors reveal
the true training-time convention (user decision 2026-09-17; the recorded
rotary_base 10000 is wrong, see sink-models/rope_report.md). Sanity-checked
in-job: target NLL on the first batch must be ~2.7, not the ~8 of the broken
spectrum.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run components/hide/typical_act_modal_C.py
"""

import modal

app = modal.App("c-components-typical-act")
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
# layer 0 was computed first (2026-09-17); layers 1-3 added later and merged
# into the same cache file (existing keys are kept, not recomputed)
MODS = [f"h.{l}.attn.{m}_proj" for l in range(4) for m in ("q", "k")]
FIRE_THRESH = 0.1


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(mods: list, log_inv_freq: list) -> dict:
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

    # install the fitted corrected RoPE spectrum (call site is inside the same
    # module, so patching the module global is seen by the model builder)
    fitted = jnp.asarray(np.exp(np.asarray(log_inv_freq, np.float64)),
                         dtype=jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: fitted

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / RUN, step=STEP, data_root=root)
    placed, ci_fn, pw = loaded.placed, loaded.ci_fn, loaded.prepared_weights
    keys = ci_fn.fn.capture_keys

    @eqx.filter_jit
    def step(placed, pw, ci_fn, tokens):
        fwd, acts = placed.component_activation_forward(pw, tokens,
                                                        capture_keys=keys)
        pre = ci_preactivations(ci_fn, select_captures(fwd.captures, keys),
                                remat=False)
        out = {}
        for m in mods:
            a = acts[m].astype(jnp.float32)                   # (B, T, C)
            ci = jnp.clip(pre[m], 0.0, 1.0).astype(jnp.float32)
            fire = ci > FIRE_THRESH
            out[m] = (jnp.sum(ci, axis=(0, 1)),
                      jnp.sum(ci * a, axis=(0, 1)),
                      jnp.sum(fire, axis=(0, 1)),
                      jnp.sum(a * fire, axis=(0, 1)))
        return out

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16

    # sanity: with the fitted spectrum the target must be good (~2.7 NLL);
    # the broken base-10000 spectrum gives ~8
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

    acc = None
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, pw, ci_fn, jnp.asarray(rows[i:i + B]))
        if acc is None:
            acc = {m: [np.zeros(v[0].shape[-1], np.float64) for _ in range(4)]
                   for m, v in out.items()}
        for m in mods:
            for k in range(4):
                acc[m][k] += np.asarray(out[m][k], dtype=np.float64)
        if (i // B) % 50 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    print(f"done, {time.time() - t0:.0f}s", flush=True)
    res = {}
    for m in mods:
        Sci, Sa_ci, F, Sa_f = acc[m]
        res[f"{m}|Sci"] = Sci
        res[f"{m}|Sa_ci"] = Sa_ci
        res[f"{m}|F"] = F
        res[f"{m}|Sa_f"] = Sa_f
    return res


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    lif = np.load(here.parents[1] / "sink-models" / "hide" / "cache"
                  / "fitted_freqs_avg.npz")["log_inv_freq"]
    (here / "cache").mkdir(exist_ok=True)
    path = here / "cache" / "typical_act_C.npz"
    old = dict(np.load(path)) if path.exists() else {}
    todo = [m for m in MODS if f"{m}|typical" not in old]
    if not todo:
        print("all requested matrices already cached; nothing to do")
        return
    res = compute.remote(todo, lif.tolist())
    for m in todo:
        with np.errstate(invalid="ignore"):
            res[f"{m}|typical"] = np.where(
                res[f"{m}|Sci"] > 0, res[f"{m}|Sa_ci"] / res[f"{m}|Sci"], 0.0)
    res = {**old, **res}
    np.savez_compressed(path, **res)
    for m in todo:
        t = res[f"{m}|typical"]
        print(f"{m}: typical act range [{t.min():.3f}, {t.max():.3f}], "
              f"median |t| {np.median(np.abs(t)):.3f}")
    print("saved", path)
