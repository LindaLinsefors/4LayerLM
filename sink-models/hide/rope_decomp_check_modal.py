"""Modal jobs answering: what RoPE did the C/D/E decompositions (and their
harvests) actually run with during training? (Follow-up to rope_report.md.)

Two probes:

1. invfreq_sink / invfreq_new -- extract the CI function's stored `inv_freq`
   rotary buffer from the decomposition checkpoints (C/D/E at the sink-loader
   commit; newA/newB at the pinned public commit as controls). The chunkwise
   transformer CI fn carries inv_freq as a stop-gradient BUFFER inside the
   checkpointed pytree (WandB logs grad norm 0 for `ci_fns.inv_freq`), so Orbax
   restores the value the INTERNAL trainer built at launch time. If the internal
   rotary generator is the non-geometric spectrum recovered in rope_report.md,
   it shows up here verbatim (head_dim 2048/16 = 128 = the target's head_dim);
   if it is textbook 10000^(-p/64) (what the public commit's init builds), the
   CI transformer at least ran textbook RoPE internally. The loaded TARGET's
   inv_freq leaves are also reported -- the public loader REBUILDS those from
   config, so they are expected textbook regardless.

2. mean_ci_sink -- per-component sample mean CI + CI>0.1 fire counts over the
   4,000 cached Pile rows under a GIVEN RoPE spectrum (fitted or configured
   base-1e4), all 24 sites, for C/D/E. Compared locally (rope_decomp_compare.py)
   against harvest.sqlite's mean_causal_importance (10M tokens, computed by the
   co-authors' internal pipeline): whichever spectrum agrees better is the one
   the internal harvest ran with.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run --detach sink-models/hide/rope_decomp_check_modal.py
Outputs: cache/ci_fn_invfreq.npz, cache/mean_ci_{fitted,configured}_{C,D,E}.npz
"""

import modal

app = modal.App("rope-decomp-check")
vol = modal.Volume.from_name("vpd-4layer")

SINK_COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"  # PR #1002 sink loader
PIN_COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"   # pinned public main

sink_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{SINK_COMMIT}"
    )
)
pin_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{PIN_COMMIT}"
    )
)

SINK_RUNS = {"C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35"}
NEW_RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
FIRE_THRESH = 0.1


def _invfreq_leaves(obj, label):
    """All array leaves whose keypath mentions inv_freq (small ones only)."""
    import jax.tree_util as jtu
    import numpy as np

    out = {}
    for path, leaf in jtu.tree_flatten_with_path(obj)[0]:
        p = jtu.keystr(path)
        if "inv_freq" in p and hasattr(leaf, "shape") and leaf.size <= 4096:
            out[f"{label}{p}"] = np.asarray(leaf, np.float64)
    return out


@app.function(image=sink_image, gpu="A10G", volumes={"/data": vol}, timeout=1800)
def invfreq_sink(run: str, step: int) -> dict:
    from pathlib import Path

    from param_decomp.experiments.lm.load_run import open_jax_run

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / run, step=step, data_root=root)
    return {**_invfreq_leaves(loaded.ci_fn, "ci_fn"),
            **_invfreq_leaves(loaded.placed, "placed")}


@app.function(image=pin_image, gpu="A10G", volumes={"/data": vol}, timeout=1800)
def invfreq_new(run: str, step: int) -> dict:
    from pathlib import Path

    from param_decomp.experiments.lm.load_run import open_jax_run

    root = Path(f"/data/new-decomps/{run}-decomposition-{step}")
    loaded = open_jax_run(root / "runs" / run, step=step, data_root=root)
    return {**_invfreq_leaves(loaded.ci_fn, "ci_fn"),
            **_invfreq_leaves(loaded.placed, "placed")}


@app.function(image=sink_image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def mean_ci_sink(run: str, log_inv_freq: list, tag: str) -> dict:
    """Sample mean CI + fire counts, all 24 sites, under the given spectrum."""
    import time
    from pathlib import Path

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run
    from param_decomp.targets import llama_simple_mlp as lsm

    spec = jnp.asarray(np.exp(np.asarray(log_inv_freq, np.float64)), jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: spec

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / run, step=100000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16

    # sanity: fitted spectrum => good target (~2.9 NLL); configured => broken (~8)
    with jax.set_mesh(loaded.mesh):
        logits = placed.clean_forward(jnp.asarray(rows[:8]), ()).output
    logits = np.asarray(logits, np.float32)
    m = logits[:, :-1].max(-1, keepdims=True)
    lse = np.log(np.exp(logits[:, :-1] - m).sum(-1)) + m[..., 0]
    tgt = rows[:8, 1:]
    nll = float((lse - np.take_along_axis(
        logits[:, :-1], tgt[..., None], -1)[..., 0]).mean())
    print(f"{run} [{tag}] target NLL (8 rows): {nll:.3f}", flush=True)
    if tag == "fitted":
        assert nll < 4.0, f"fitted RoPE patch did not take effect (NLL {nll:.2f})"
    else:
        assert nll > 6.0, f"expected broken configured forward, got NLL {nll:.2f}"

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        fwd = placed.clean_forward(tokens, keys)
        pre = ci_preactivations(ci_fn, select_captures(fwd.captures, keys),
                                remat=False)
        out = {}
        for mname, p in pre.items():
            ci = jnp.clip(p, 0.0, 1.0).astype(jnp.float32)
            out[mname] = (jnp.sum(ci, axis=(0, 1)),
                          jnp.sum(ci > FIRE_THRESH, axis=(0, 1)))
        return out

    acc = {}
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        for mname, (s, f) in out.items():
            if mname not in acc:
                acc[mname] = [np.zeros(s.shape[-1], np.float64),
                              np.zeros(s.shape[-1], np.float64)]
            acc[mname][0] += np.asarray(s, np.float64)
            acc[mname][1] += np.asarray(f, np.float64)
        if (i // B) % 50 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    res = {"nll": np.array(nll), "T": np.array(rows.size, np.float64)}
    for mname, (s, f) in acc.items():
        res[f"{mname}|Sci"] = s
        res[f"{mname}|F"] = f
    return res


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    cache = here / "cache"
    lif_fitted = np.load(cache / "fitted_freqs_avg.npz")["log_inv_freq"]
    lif_cfg = -(np.arange(64) / 64) * np.log(1e4)

    hs = {name: invfreq_sink.spawn(run, 100000)
          for name, run in SINK_RUNS.items()}
    hs |= {name: invfreq_new.spawn(run, 800000)
           for name, run in NEW_RUNS.items()}
    hm = {}
    for name, run in SINK_RUNS.items():
        for tag, lif in [("fitted", lif_fitted), ("configured", lif_cfg)]:
            hm[(name, tag)] = mean_ci_sink.spawn(run, lif.tolist(), tag)

    inv = {}
    for name, h in hs.items():
        for k, v in h.get().items():
            inv[f"{name}|{k}"] = v
        print(f"{name}: {[k for k in h.get()]}")
    np.savez(cache / "ci_fn_invfreq.npz", **inv)
    print("saved", cache / "ci_fn_invfreq.npz")

    for (name, tag), h in hm.items():
        res = h.get()
        np.savez_compressed(cache / f"mean_ci_{tag}_{name}.npz", **res)
        print(f"saved mean_ci_{tag}_{name}.npz  (target NLL {float(res['nll']):.3f})")
