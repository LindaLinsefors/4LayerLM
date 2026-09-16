"""Modal GPU job: per-component activation-sign statistics for all three
decompositions (old = s-55ea3f9b via Torch, newA/newB via JAX), over the same
4,000 cached Pile rows.

Purpose: fix each component's sign gauge (U_c, V_c) -> (-U_c, -V_c) so that the
component activation a_c = V_c . x is POSITIVE on most tokens where the
component is causally important (CI > 0.1) — needed to draw SIGNED cos(U)/
cos(V) heatmaps. Per module and component this job records:
  Ssum = sum of (x @ V)_c over tokens with CI_c > 0.1
  Npos = # of those tokens with (x @ V)_c > 0
  F    = # of those tokens
  Sall = sum of (x @ V)_c over ALL tokens (fallback for never-firing comps)
The sign convention itself (majority of firing tokens positive; ties by Ssum;
F = 0 falls back to Sall) is applied at build time in interactive_cross.py.

Run:  modal run coci-heatmaps/hide/act_signs_modal.py
Then: modal volume get vpd-4layer /act_signs_<name>.npz coci-heatmaps/hide/cache/
Output npz keys per module: "<mod>|Ssum"/"<mod>|Sall" (C, float64),
"<mod>|Npos"/"<mod>|F" (C, int64).
"""

import modal

app = modal.App("act-signs")
vol = modal.Volume.from_name("vpd-4layer")

torch_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)
COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
jax_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
FIRE_THRESH = 0.1
MODULES = [f"h.{l}.{m}" for l in range(4)
           for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                     "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]


@app.function(image=torch_image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute_old() -> None:
    import os
    import time
    from pathlib import Path

    os.environ["PARAM_DECOMP_OUT_DIR"] = "/data/param_decomp_out"

    import numpy as np
    import torch

    import param_decomp.pretrain.run_info as pri

    def _local_files(entity: str, project: str, run_id: str) -> pri.WandbDownloadedFiles:
        d = Path(f"/data/param_decomp_out/pretrain_cache/{project}-{run_id}")
        assert d.exists(), d
        return pri.WandbDownloadedFiles(
            checkpoint=d / "model_step_99999.pt",
            config=d / "final_config.yaml",
            model_config=d / "model_config.yaml",
            tokenizer=d / "tokenizer.json",
        )

    pri._download_wandb_files = _local_files

    from param_decomp.models.component_model import ComponentModel

    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to(device)
    params = dict(model.named_parameters())
    V = {m: params[f"_components.{m.replace('.', '-')}.V"] for m in MODULES}

    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")])
    acc = {m: [torch.zeros(V[m].shape[1], device=device, dtype=torch.float64)
               for _ in range(4)] for m in MODULES}  # Ssum, Npos, F, Sall

    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(rows), 16):
            tok = rows[b:b + 16].to(device)
            out = model(tok, cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            for m in MODULES:
                a = (out.cache[m] @ V[m]).flatten(0, 1)          # (B*S, C)
                fire = ci_out.lower_leaky[m].flatten(0, 1) > FIRE_THRESH
                acc[m][0] += (a * fire).sum(0).double()
                acc[m][1] += (fire & (a > 0)).sum(0).double()
                acc[m][2] += fire.sum(0).double()
                acc[m][3] += a.sum(0).double()
            if (b // 16) % 25 == 0:
                print(f"batch {b // 16 + 1}/{len(rows) // 16}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for m in MODULES:
        data[f"{m}|Ssum"] = acc[m][0].cpu().numpy()
        data[f"{m}|Npos"] = acc[m][1].cpu().numpy().astype(np.int64)
        data[f"{m}|F"] = acc[m][2].cpu().numpy().astype(np.int64)
        data[f"{m}|Sall"] = acc[m][3].cpu().numpy()
    np.savez_compressed("/data/act_signs_old.npz", **data)
    vol.commit()
    print(f"saved /data/act_signs_old.npz ({time.time() - t0:.0f}s total)")


@app.function(image=jax_image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute_new(name: str) -> None:
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

    run = RUNS[name]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / run, step=800000, data_root=root)
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
            a = acts[s].astype(jnp.float32)                       # (B, T, C)
            fire = jnp.clip(pre[s], 0.0, 1.0) > FIRE_THRESH
            out[s] = (jnp.sum(a * fire, axis=(0, 1)),
                      jnp.sum(fire & (a > 0), axis=(0, 1)),
                      jnp.sum(fire, axis=(0, 1)),
                      jnp.sum(a, axis=(0, 1)))
        return out

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
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

    assert set(sites) == set(MODULES)
    data = {}
    for s in sites:
        data[f"{s}|Ssum"] = acc[s][0]
        data[f"{s}|Npos"] = acc[s][1].astype(np.int64)
        data[f"{s}|F"] = acc[s][2].astype(np.int64)
        data[f"{s}|Sall"] = acc[s][3]
    np.savez_compressed(f"/data/act_signs_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/act_signs_{name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(compute_old.remote),
                ex.submit(compute_new.remote, "newA"),
                ex.submit(compute_new.remote, "newB")]
        for f in futs:
            f.result()
