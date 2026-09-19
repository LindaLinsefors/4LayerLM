"""Modal GPU jobs: per-decomposition statistics for decomposition F =
p-c45e0001 (the corrected-RoPE re-decomposition of C's target t-87f91319,
trained by sink-models/redo-decomps/) — the same four caches
compare-decomps/hide/sink_stats_modal.py builds for C/D/E:

  coci       -> /coci_F.npz        (per-matrix co-CI r f16, mean, F, T)
  top_tokens -> /top_tokens_F.npz  (top-20 input tokens by summed CI)
  act_signs  -> /act_signs_F.npz   (activation-sign stats for the signed gauge)
  uv         -> /uv_F.npz          (U/V factors f16 — ALL components, id order)

All over the same 4,000 cached Pile rows (/pile_rows.npy, batches of 16).
RoPE: unlike C/D/E, F was TRAINED with the fitted corrected spectrum
(sink-models/hide/cache/fitted_freqs_avg.npz, monkeypatched into
plain_rope_inv_freq by redo-decomps/hide/train_modal.py), so every F forward
must install the same patch — no two-regime split for F. Sanity-checked
in-job: target NLL ~2.7 under the patch (the un-patched base-1e4 forward
would give ~8, but that would be the WRONG model for F).

F's run dir has no deliverable.yaml; open_jax_run falls back to the pinned
launch_config.yaml, which resolves fine at the sink PR commit.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run components/hide/f_stats_modal.py
Then: modal volume get vpd-4layer /coci_F.npz coci-heatmaps/hide/cache/
      modal volume get vpd-4layer /uv_F.npz /act_signs_F.npz compare-decomps/hide/cache/
      modal volume get vpd-4layer /top_tokens_F.npz mean-ci-widget/hide/cache/
"""

import modal

app = modal.App("components-f-stats")
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
FIRE_THRESH = 0.1
VOCAB = 50277
K = 20
DIMS = {"attn.q_proj": (768, 768), "attn.k_proj": (768, 768),
        "attn.v_proj": (768, 768), "attn.o_proj": (768, 768),
        "mlp.c_fc": (768, 3072), "mlp.down_proj": (3072, 768)}  # (d_in, d_out)


def _load(inv_freq):
    """Install the fitted RoPE (F's training-time spectrum), open the run,
    and assert the target forward is healthy under it."""
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp

    from param_decomp.targets import llama_simple_mlp as lsm
    from param_decomp.experiments.lm.load_run import open_jax_run

    spec = jnp.asarray(np.asarray(inv_freq, np.float64), jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: spec

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / RUN, step=STEP, data_root=root)

    rows = np.load("/data/pile_rows.npy")
    keys = loaded.ci_fn.fn.capture_keys
    with jax.set_mesh(loaded.mesh):
        logits = loaded.placed.clean_forward(jnp.asarray(rows[:8]), keys).output
    logits = np.asarray(logits, np.float32)
    lse = np.log(np.exp(logits[:, :-1] - logits[:, :-1].max(-1, keepdims=True)
                        ).sum(-1)) + logits[:, :-1].max(-1)
    tgt = rows[:8, 1:]
    nll = float((lse - np.take_along_axis(
        logits[:, :-1], tgt[..., None], -1)[..., 0]).mean())
    print(f"F target NLL (first 8 rows, fitted RoPE): {nll:.3f}", flush=True)
    assert nll < 4.0, f"RoPE patch did not take effect (NLL {nll:.2f})"
    return loaded, rows


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def coci(inv_freq: list) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from jax.sharding import PartitionSpec as P

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    loaded, rows = _load(inv_freq)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"F = {RUN}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        out = {}
        for s in sites:
            c = jnp.clip(pre[s], 0.0, 1.0).reshape(-1, pre[s].shape[-1])
            g = jnp.einsum("tc,td->cd", c, c, out_sharding=P(None, None))
            out[s] = (c.sum(0), (c > FIRE_THRESH).sum(0), g)
        return out

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
    np.savez("/data/coci_F.npz", **data)
    vol.commit()
    print(f"saved /data/coci_F.npz ({time.time() - t0:.0f}s total)")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def top_tokens(inv_freq: list) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    loaded, rows = _load(inv_freq)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"F = {RUN}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {s: jnp.clip(pre[s], 0.0, 1.0)
                     .reshape(-1, pre[s].shape[-1]).astype(jnp.float16)
                for s in sites}

    scatter = jax.jit(lambda a, c, t: a.at[t].add(c), donate_argnums=(0,))

    B = 16
    acc = None
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        out = {s: np.asarray(v) for s, v in out.items()}
        if acc is None:
            acc = {s: jnp.zeros((VOCAB, v.shape[1]), jnp.float32)
                   for s, v in out.items()}
        tok = jnp.asarray(rows[i:i + B].reshape(-1))
        for s in sites:
            acc[s] = scatter(acc[s], jnp.asarray(out[s], jnp.float32), tok)
        if (i // B) % 25 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for s in sites:
        A = np.asarray(acc.pop(s))
        top = np.argpartition(-A, K, axis=0)[:K].T
        vals = np.take_along_axis(A.T, top, axis=1)
        order = np.argsort(-vals, axis=1)
        data[f"{s}|top_ids"] = np.take_along_axis(top, order, axis=1).astype(np.int32)
        data[f"{s}|top_ci"] = np.take_along_axis(vals, order, axis=1).astype(np.float32)
        data[f"{s}|total"] = A.sum(0, dtype=np.float64)
    np.savez_compressed("/data/top_tokens_F.npz", **data)
    vol.commit()
    print(f"saved /data/top_tokens_F.npz ({time.time() - t0:.0f}s total)")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def act_signs(inv_freq: list) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    loaded, rows = _load(inv_freq)
    placed, ci_fn, pw = loaded.placed, loaded.ci_fn, loaded.prepared_weights
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"F = {RUN}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, pw, ci_fn, tokens):
        fwd, acts = placed.component_activation_forward(pw, tokens,
                                                        capture_keys=keys)
        pre = ci_preactivations(ci_fn, select_captures(fwd.captures, keys),
                                remat=False)
        out = {}
        for s in sites:
            a = acts[s].astype(jnp.float32)
            fire = jnp.clip(pre[s], 0.0, 1.0) > FIRE_THRESH
            out[s] = (jnp.sum(a * fire, axis=(0, 1)),
                      jnp.sum(fire & (a > 0), axis=(0, 1)),
                      jnp.sum(fire, axis=(0, 1)),
                      jnp.sum(a, axis=(0, 1)))
        return out

    B = 16
    acc = None
    t0 = time.time()
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, pw, ci_fn, jnp.asarray(rows[i:i + B]))
        if acc is None:
            acc = {s: [np.zeros(v[0].shape[-1], np.float64) for _ in range(4)]
                   for s, v in out.items()}
        for s in sites:
            for k in range(4):
                acc[s][k] += np.asarray(out[s][k], dtype=np.float64)
        if (i // B) % 25 == 0:
            print(f"batch {i // B + 1}/{len(rows) // B}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for s in sites:
        data[f"{s}|Ssum"] = acc[s][0]
        data[f"{s}|Npos"] = acc[s][1].astype(np.int64)
        data[f"{s}|F"] = acc[s][2].astype(np.int64)
        data[f"{s}|Sall"] = acc[s][3]
    np.savez_compressed("/data/act_signs_F.npz", **data)
    vol.commit()
    print(f"saved /data/act_signs_F.npz ({time.time() - t0:.0f}s total)")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=1800)
def uv(inv_freq: list) -> None:
    import numpy as np

    loaded, _ = _load(inv_freq)
    pw = loaded.prepared_weights
    sites = list(loaded.ci_fn.fn.output_names)
    print(f"F = {RUN}: kinds {sorted(pw)}", flush=True)

    data = {}
    for site in sites:  # "h.<l>.<attn|mlp>.<kind>"
        _, l, mod = site.split(".", 2)
        layer = int(l)
        kind = mod.split(".")[-1]
        assert kind in pw, f"{kind} not in {sorted(pw)}"
        V = np.asarray(pw[kind]["V"][layer], np.float32)  # (d_in, C)
        U = np.asarray(pw[kind]["U"][layer], np.float32)  # (C, d_out)
        d_in, d_out = DIMS[mod]
        assert V.shape[0] == d_in and U.shape[1] == d_out and \
            V.shape[1] == U.shape[0], f"{site}: V {V.shape}, U {U.shape}"
        data[f"{site}|V"] = V.T.astype(np.float16)   # (C, d_in) — ALL comps
        data[f"{site}|U"] = U.astype(np.float16)     # (C, d_out)
    np.savez("/data/uv_F.npz", **data)
    vol.commit()
    print("saved /data/uv_F.npz")


@app.local_entrypoint()
def main() -> None:
    import concurrent.futures as cf
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    lif = np.exp(np.load(here.parents[1] / "sink-models" / "hide" / "cache"
                         / "fitted_freqs_avg.npz")["log_inv_freq"]).tolist()
    with cf.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(fn.remote, lif)
                for fn in (coci, top_tokens, act_signs, uv)]
        for f in futs:
            f.result()
