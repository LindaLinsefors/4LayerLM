"""Modal GPU jobs: per-decomposition statistics for the attention-sink-model
decompositions C = p-d60af588, D = p-fecd6a6b, E = p-bd411e35 (see CLAUDE.md
"Attention-sink models & decompositions") and F = p-c45e0001 (the
corrected-RoPE re-decomposition of C's target, sink-models/redo-decomps) —
everything the compare-decomps widget needs per decomposition, mirroring the
newA/newB caches:

  coci       -> /coci_<name>.npz        (per-matrix co-CI r f16, mean, F, T)
  top_tokens -> /top_tokens_<name>.npz  (top-20 input tokens by summed CI)
  act_signs  -> /act_signs_<name>.npz   (activation-sign stats for the
                                         signed-cos gauge, CI > 0.1 tokens)
  uv         -> /uv_<name>.npz          (U/V factors f16 — ALL components,
                                         unlike the alive-only uv_newA/newB)
  pos_fires  -> /pos_fires_sink_<name>.npz  (per-position CI>0.1 fire counts
                                         + CI sums; F only — C/D/E already
                                         have theirs from sink-models/hide/
                                         pos_fires_sink_modal.py)

All over the same 4,000 cached Pile rows (/pile_rows.npy, batches of 16) as
every other coci/compare cache. Loader = the sink PR commit (82a67f71c);
C/D/E archives already extracted on the volume by sink-models/hide/
pos_fires_sink_modal.py's fetch stage; F's run dir was written in place by
its trainer (sink-models/redo-decomps/hide/train_modal.py).

RoPE: C/D/E forwards use the loader as-is (configured base-1e4 — what those
decompositions were trained against, see the sink-models ⚠ section). F was
TRAINED with the fitted corrected spectrum monkeypatched in, so its forward
installs the same patch (/fitted_freqs_avg.npz on the volume) + an in-job
NLL sanity assert.

Run:  modal run compare-decomps/hide/sink_stats_modal.py --names F
Then: modal volume get vpd-4layer /<file> compare-decomps/hide/cache/
"""

import modal

app = modal.App("compare-decomps-sink-stats")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35",
        "F": "p-c45e0001"}
FITTED_ROPE = {"F"}  # decompositions trained with the fitted corrected RoPE
STEP = 100000
FIRE_THRESH = 0.1
VOCAB = 50277
K = 20
DIMS = {"attn.q_proj": (768, 768), "attn.k_proj": (768, 768),
        "attn.v_proj": (768, 768), "attn.o_proj": (768, 768),
        "mlp.c_fc": (768, 3072), "mlp.down_proj": (3072, 768)}  # (d_in, d_out)


def _load(name):
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp

    from param_decomp.experiments.lm.load_run import open_jax_run

    run = RUNS[name]
    root = Path("/data/sink-models")
    if name in FITTED_ROPE:
        # F was trained with this exact spectrum monkeypatched in (see
        # sink-models/redo-decomps/hide/train_modal.py); install it before the
        # target is built. Patch the module global — the builder reads it.
        from param_decomp.targets import llama_simple_mlp as lsm
        lif = np.load("/data/fitted_freqs_avg.npz")["log_inv_freq"]
        spec = jnp.asarray(np.exp(np.asarray(lif, np.float64)), jnp.float32)
        lsm.plain_rope_inv_freq = lambda cfg: spec
    loaded = open_jax_run(root / "runs" / run, step=STEP, data_root=root)
    if name in FITTED_ROPE:
        # sanity: fitted spectrum ⇒ target NLL ~2.5; broken base-1e4 gives ~8
        rows = np.load("/data/pile_rows.npy")[:8]
        keys = loaded.ci_fn.fn.capture_keys
        with jax.set_mesh(loaded.mesh):
            logits = np.asarray(
                loaded.placed.clean_forward(jnp.asarray(rows), keys).output,
                np.float32)
        mx = logits[:, :-1].max(-1, keepdims=True)
        lse = np.log(np.exp(logits[:, :-1] - mx).sum(-1)) + mx[..., 0]
        nll = float((lse - np.take_along_axis(
            logits[:, :-1], rows[:, 1:, None], -1)[..., 0]).mean())
        print(f"{name} target NLL (first 8 rows, fitted RoPE): {nll:.3f}",
              flush=True)
        assert nll < 4.0, f"RoPE patch did not take effect (NLL {nll:.2f})"
    return run, loaded


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def coci(name: str) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from jax.sharding import PartitionSpec as P

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    run, loaded = _load(name)
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
            g = jnp.einsum("tc,td->cd", c, c, out_sharding=P(None, None))
            out[s] = (c.sum(0), (c > FIRE_THRESH).sum(0), g)
        return out

    rows = np.load("/data/pile_rows.npy")
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
    np.savez(f"/data/coci_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/coci_{name}.npz ({time.time() - t0:.0f}s total)")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def top_tokens(name: str) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    run, loaded = _load(name)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"{name} = {run}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {s: jnp.clip(pre[s], 0.0, 1.0)
                     .reshape(-1, pre[s].shape[-1]).astype(jnp.float16)
                for s in sites}

    scatter = jax.jit(lambda a, c, t: a.at[t].add(c), donate_argnums=(0,))

    rows = np.load("/data/pile_rows.npy")
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
    np.savez_compressed(f"/data/top_tokens_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/top_tokens_{name}.npz ({time.time() - t0:.0f}s total)")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def act_signs(name: str) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    run, loaded = _load(name)
    placed, ci_fn, pw = loaded.placed, loaded.ci_fn, loaded.prepared_weights
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"{name} = {run}: {len(sites)} sites", flush=True)

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

    rows = np.load("/data/pile_rows.npy")
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
    np.savez_compressed(f"/data/act_signs_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/act_signs_{name}.npz ({time.time() - t0:.0f}s total)")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=1800)
def uv(name: str) -> None:
    import numpy as np

    run, loaded = _load(name)
    pw = loaded.prepared_weights
    sites = list(loaded.ci_fn.fn.output_names)
    print(f"{name} = {run}: kinds {sorted(pw)}", flush=True)

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
    np.savez(f"/data/uv_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/uv_{name}.npz")


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def pos_fires(name: str) -> None:
    """Per-position CI>0.1 fire counts + CI sums (the pos_fires_sink format:
    "<mod>|Fp" (C, 512) int32, "<mod>|Sp" (C, 512) float32)."""
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures

    run, loaded = _load(name)
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
            c = jnp.clip(pre[s], 0.0, 1.0)          # (B, T, C)
            out[s] = ((c > FIRE_THRESH).sum(0).T,   # (C, T)
                      c.sum(0).T)
        return out

    rows = np.load("/data/pile_rows.npy")
    B = 16
    Fp = Sp = None
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            out = step(placed, ci_fn, jnp.asarray(rows[i:i + B]))
            if Fp is None:
                Fp = {s: np.zeros(out[s][0].shape, np.int64) for s in sites}
                Sp = {s: np.zeros(out[s][1].shape, np.float64) for s in sites}
            for s in sites:
                Fp[s] += np.asarray(out[s][0], np.int64)
                Sp[s] += np.asarray(out[s][1], np.float64)
            if (i // B) % 25 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for s in sites:
        data[f"{s}|Fp"] = Fp[s].astype(np.int32)
        data[f"{s}|Sp"] = Sp[s].astype(np.float32)
    np.savez_compressed(f"/data/pos_fires_sink_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/pos_fires_sink_{name}.npz ({time.time() - t0:.0f}s)")


@app.local_entrypoint()
def main(names: str = "C,D,E") -> None:
    import concurrent.futures as cf
    todo = names.split(",")
    fns = [coci, top_tokens, act_signs, uv]
    if todo == ["F"]:
        fns.append(pos_fires)
    with cf.ThreadPoolExecutor(20) as ex:
        futs = [ex.submit(fn.remote, n) for fn in fns for n in todo]
        for f in futs:
            f.result()
