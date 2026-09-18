"""Modal GPU job: activation examples for the sink decompositions C / D / E.

For every ALIVE component (sample mean CI > 1e-6, the established alive proxy
for these runs — alive index sets are passed in from the local
coci-heatmaps/hide/cache/coci_{C,D,E}.npz so the widget matches the other
C/D/E analyses exactly): the top up-to-16 activating windows over the 4,000
cached Pile rows (2.05M tokens), with a 51-token CI trace around each fire.

CI = clip(preactivations, 0, 1), identical to coci_compute_sink_modal.py
(⚠ same public-loader broken-RoPE caveat, sink-models/rope_report.md).

Pass 1: per batch, per site, GPU top-64 CI positions per component; merged on
        the host into a running per-component top-64 candidate list.
Pass 2: winners chosen (greedy non-overlap: same row & |Δpos| < 41 skipped),
        then one more sweep gathers each winner's 51-token CI trace
        (window [pos-25, pos+26) clipped to the 512-token row).

Output per run: /data/examples_{C,D,E}.npz with, per site:
  "<site>|comps"  int32 (n,)        alive component ids (kept order)
  "<site>|nwin"   uint8 (n,)        number of stored windows per component
  "<site>|row"    uint16 (n,16)     cached-row index of each window
  "<site>|pos"    uint16 (n,16)     fire position within the row
  "<site>|wstart" uint16 (n,16)     window start within the row (51 tokens)
  "<site>|trace"  float16 (n,16,51) CI trace over the window
Unused window slots have row = 65535.

Run:  modal run sink-harvest-widget/hide/examples_compute_modal.py
Then: modal volume get vpd-4layer /examples_C.npz sink-harvest-widget/hide/cache/  (etc.)
"""

import modal

app = modal.App("sink-activation-examples")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"  # PR #1002 sink loader
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35"}
STEP = 100000
N_WIN = 16       # windows kept per component
N_CAND = 64      # top-CI candidates tracked per component
HALF = 25        # context tokens each side of the fire (window = 51)
ROW_LEN = 512


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def compute(name: str, alive: dict) -> None:
    """alive: site -> int32 array of alive component ids (from the local
    coci caches, so the alive sets match the rest of the project)."""
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

    run = RUNS[name]
    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / run, step=STEP, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"{name} = {run}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step_topk(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        out = {}
        for s in sites:
            c = jnp.clip(pre[s], 0.0, 1.0).reshape(-1, pre[s].shape[-1])
            # token axis is mesh-sharded; top_k needs it replicated
            ct = jax.sharding.reshard(c.T, P(None, None))
            vals, idx = jax.lax.top_k(ct, N_CAND)   # (C, N_CAND)
            out[s] = (vals.astype(jnp.float16), idx.astype(jnp.int32))
        return out

    @eqx.filter_jit
    def step_full(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {s: jnp.clip(pre[s], 0.0, 1.0).astype(jnp.float16)
                for s in sites}

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    t0 = time.time()

    # ---- pass 1: running top-N_CAND (val, global flat pos) per component
    cand_v = cand_p = None
    for i in range(0, len(rows), B):
        with jax.set_mesh(loaded.mesh):
            out = step_topk(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        out = {s: (np.asarray(v[0], np.float32), np.asarray(v[1]))
               for s, v in out.items()}
        if cand_v is None:
            cand_v = {s: np.full((o[0].shape[0], N_CAND), -1, np.float32)
                      for s, o in out.items()}
            cand_p = {s: np.zeros(cand_v[s].shape, np.int64) for s in sites}
        for s in sites:
            bv, bp = out[s][0], out[s][1].astype(np.int64) + i * ROW_LEN
            v = np.concatenate([cand_v[s], bv], axis=1)
            p = np.concatenate([cand_p[s], bp], axis=1)
            sel = np.argpartition(-v, N_CAND - 1, axis=1)[:, :N_CAND]
            cand_v[s] = np.take_along_axis(v, sel, axis=1)
            cand_p[s] = np.take_along_axis(p, sel, axis=1)
        if (i // B) % 50 == 0:
            print(f"pass1 batch {i // B + 1}/{len(rows) // B} "
                  f"{time.time() - t0:.0f}s", flush=True)

    # ---- choose winners: alive comps only, greedy non-overlapping windows
    chosen = {}   # site -> dict comp -> list[(row, pos)]
    need = {}     # site -> dict row -> list[(comp_i, win_i, wstart)]
    result = {}
    for s in sites:
        ids = np.asarray(alive[s], np.int64)
        n = len(ids)
        nwin = np.zeros(n, np.uint8)
        rowa = np.full((n, N_WIN), 65535, np.uint16)
        posa = np.zeros((n, N_WIN), np.uint16)
        wsta = np.zeros((n, N_WIN), np.uint16)
        need_s = {}
        for ci_i, comp in enumerate(ids):
            order = np.argsort(-cand_v[s][comp])
            kept = []
            for j in order:
                v = float(cand_v[s][comp, j])
                if v <= 1e-6 or len(kept) >= N_WIN:
                    break
                g = int(cand_p[s][comp, j])
                r, p = g // ROW_LEN, g % ROW_LEN
                if any(r == kr and abs(p - kp) < 41 for kr, kp in kept):
                    continue
                kept.append((r, p))
            for w, (r, p) in enumerate(kept):
                ws = min(max(p - HALF, 0), ROW_LEN - (2 * HALF + 1))
                rowa[ci_i, w], posa[ci_i, w], wsta[ci_i, w] = r, p, ws
                need_s.setdefault(r, []).append((ci_i, w, ws, int(comp)))
            nwin[ci_i] = len(kept)
        result[f"{s}|comps"] = ids.astype(np.int32)
        result[f"{s}|nwin"] = nwin
        result[f"{s}|row"] = rowa
        result[f"{s}|pos"] = posa
        result[f"{s}|wstart"] = wsta
        result[f"{s}|trace"] = np.zeros((n, N_WIN, 2 * HALF + 1), np.float16)
        need[s] = need_s
    print(f"winners chosen {time.time() - t0:.0f}s", flush=True)

    # ---- pass 2: gather CI traces for winner windows
    needed_batches = sorted({r - r % B for s in sites for r in need[s]})
    for bi, i in enumerate(needed_batches):
        with jax.set_mesh(loaded.mesh):
            out = step_full(placed, ci_fn, jnp.asarray(rows[i:i + B]))
        for s in sites:
            got = None
            for r in range(i, i + B):
                if r not in need[s]:
                    continue
                if got is None:
                    got = np.asarray(out[s], np.float16)  # (B, 512, C)
                for ci_i, w, ws, comp in need[s][r]:
                    result[f"{s}|trace"][ci_i, w] = \
                        got[r - i, ws:ws + 2 * HALF + 1, comp]
        if bi % 50 == 0:
            print(f"pass2 batch {bi + 1}/{len(needed_batches)} "
                  f"{time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"/data/examples_{name}.npz", **result)
    vol.commit()
    print(f"saved /data/examples_{name}.npz ({time.time() - t0:.0f}s total)",
          flush=True)


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    root = Path(__file__).resolve().parent.parent.parent
    alive_by_run = {}
    for name in RUNS:
        z = np.load(root / "coci-heatmaps" / "hide" / "cache"
                    / f"coci_{name}.npz")
        alive_by_run[name] = {
            k[:-5]: np.nonzero(z[k].astype(float) > 1e-6)[0].astype(np.int32)
            for k in z.files if k.endswith("|mean")}
        print(name, "alive:",
              sum(len(v) for v in alive_by_run[name].values()))
    list(compute.starmap([(n, alive_by_run[n]) for n in RUNS]))


if __name__ == "__main__":
    main()
