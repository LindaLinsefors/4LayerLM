"""Modal GPU job: component-activation traces for the activation-examples
widget (C / D / E) — the companion of examples_compute_modal.py.

For every example window already chosen by examples_compute_modal.py (local
hide/cache/examples_{C,D,E}.npz, passed in as arguments), gather the 51-token
trace of the RAW input activation a_t = x_t · V_c at the component's site
(same quantity as the act_signs / cos-sim sign-gauge work; the majority-
positive sign gauge is applied at build time from
compare-decomps/hide/cache/act_signs_{C,D,E}.npz — raw values stored here).

One sweep over the needed 16-row batches per run via
placed.component_activation_forward (⚠ same public-loader broken-RoPE caveat
as the CI traces, sink-models/rope_report.md).

Output per run: /data/examples_act_{C,D,E}.npz with, per site:
  "<site>|act"  float16 (n, 16, 51)  raw x·V trace, same window slots as
                                     examples_{name}.npz (unused slots 0)

Run:  modal run sink-harvest-widget/hide/examples_act_modal.py
Then: modal volume get vpd-4layer /examples_act_C.npz sink-harvest-widget/hide/cache/  (etc.)
"""

import modal

app = modal.App("sink-activation-example-acts")
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
N_WIN = 16
W = 51
ROW_LEN = 512


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def compute(name: str, wins: dict) -> None:
    """wins: site -> dict(comps int32 (n,), nwin uint8 (n,),
    row uint16 (n,16), wstart uint16 (n,16)) from the local examples cache."""
    import time
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.experiments.lm.load_run import open_jax_run

    run = RUNS[name]
    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / run, step=STEP, data_root=root)
    placed, ci_fn, pw = loaded.placed, loaded.ci_fn, loaded.prepared_weights
    keys = ci_fn.fn.capture_keys
    sites = list(ci_fn.fn.output_names)
    print(f"{name} = {run}: {len(sites)} sites", flush=True)

    @eqx.filter_jit
    def step(placed, pw, tokens):
        _, acts = placed.component_activation_forward(pw, tokens,
                                                      capture_keys=keys)
        return {s: acts[s].astype(jnp.float16) for s in sites}

    # need: site -> row -> list[(comp_i, win_i, wstart, comp)]
    need, result = {}, {}
    for s in sites:
        w = wins[s]
        n = len(w["comps"])
        result[f"{s}|act"] = np.zeros((n, N_WIN, W), np.float16)
        need_s = {}
        for ci_i in range(n):
            for wi in range(int(w["nwin"][ci_i])):
                r = int(w["row"][ci_i, wi])
                need_s.setdefault(r, []).append(
                    (ci_i, wi, int(w["wstart"][ci_i, wi]),
                     int(w["comps"][ci_i])))
        need[s] = need_s

    rows = np.load("/data/pile_rows.npy")
    B = 16
    t0 = time.time()
    needed_batches = sorted({r - r % B for s in sites for r in need[s]})
    for bi, i in enumerate(needed_batches):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, pw, jnp.asarray(rows[i:i + B]))
        for s in sites:
            got = None
            for r in range(i, i + B):
                if r not in need[s]:
                    continue
                if got is None:
                    got = np.asarray(out[s], np.float16)  # (B, 512, C)
                for ci_i, wi, ws, comp in need[s][r]:
                    result[f"{s}|act"][ci_i, wi] = got[r - i, ws:ws + W, comp]
        if bi % 50 == 0:
            print(f"batch {bi + 1}/{len(needed_batches)} "
                  f"{time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"/data/examples_act_{name}.npz", **result)
    vol.commit()
    print(f"saved /data/examples_act_{name}.npz "
          f"({time.time() - t0:.0f}s total)", flush=True)


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    root = Path(__file__).resolve().parent.parent.parent
    args = []
    for name in RUNS:
        ex = np.load(root / "sink-harvest-widget" / "hide" / "cache"
                     / f"examples_{name}.npz")
        wins = {}
        for k in ex.files:
            if k.endswith("|comps"):
                s = k[:-6]
                wins[s] = {"comps": ex[f"{s}|comps"], "nwin": ex[f"{s}|nwin"],
                           "row": ex[f"{s}|row"], "wstart": ex[f"{s}|wstart"]}
        print(name, "windows:",
              sum(int(w["nwin"].sum()) for w in wins.values()))
        args.append((name, wins))
    list(compute.starmap(args))


if __name__ == "__main__":
    main()
