"""Modal GPU job: fire-site statistics for ALL >=2-member alive co-CI>0.9
clusters of the four h.0.attn matrices — old decomposition (s-55ea3f9b, Torch
ComponentModel path as in coci_compute_modal.py) and newA/newB (JAX path as in
big_clusters_modal.py). Cluster member ids from cache/l0_attn_clusters.json
(l0_attn_clusters_extract.py).

Per selected component: per-position CI>0.1 fire counts (512) and fires-on-EOS
count; per cluster: full fired-token histogram. Same 4,000 cached Pile rows.

Run:  modal run coci-heatmaps/hide/l0_attn_clusters_modal.py
Then: modal volume get vpd-4layer /l0_attn_clusters_<name>.npz coci-heatmaps/hide/cache/
      (<name> = old, newA, newB)
Npz keys per site: "<site>|ids", "<site>|pos_counts" (n,512), "<site>|eos" (n,);
per cluster k of a site: "<site>#<k>|ids", "<site>#<k>|tok" (vocab histogram).
"""

import json
from pathlib import Path

import modal

app = modal.App("l0-attn-clusters")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
jax_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)
torch_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)

RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
FIRE_THRESH = 0.1
VOCAB = 50277
EOS = 0


def _accumulate(sites, sel, fires_np, tok_np, pos_counts, eos_counts, tok_counts):
    """Shared host-side accumulation; fires_np: {site: (B, S, n_sel) bool}."""
    import numpy as np

    for s, cls in sites.items():
        fires = fires_np[s]
        pos_counts[s] += fires.sum(0).T
        eos_counts[s] += fires[tok_np == EOS].sum(0)
        off = 0
        for k, c in enumerate(cls):
            w = fires[:, :, off:off + len(c)].sum(-1)
            tok_counts[s][k] += np.bincount(
                tok_np.ravel(), weights=w.ravel(), minlength=VOCAB
            ).astype(np.int64)
            off += len(c)


def _save(name, sites, sel, pos_counts, eos_counts, tok_counts):
    import numpy as np

    data = {}
    for s, cls in sites.items():
        data[f"{s}|ids"] = sel[s]
        data[f"{s}|pos_counts"] = pos_counts[s]
        data[f"{s}|eos"] = eos_counts[s]
        for k, c in enumerate(cls):
            data[f"{s}#{k}|ids"] = np.array(c)
            data[f"{s}#{k}|tok"] = tok_counts[s][k]
    np.savez_compressed(f"/data/l0_attn_clusters_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/l0_attn_clusters_{name}.npz")


def _site_groups(clusters):
    import numpy as np

    sites: dict[str, list] = {}
    for cl in clusters:
        sites.setdefault(cl["site"], []).append(cl["ids"])
    sel = {s: np.concatenate([np.array(c) for c in cls])
           for s, cls in sites.items()}
    return sites, sel


@app.function(image=torch_image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute_old(clusters: list[dict]) -> None:
    import os
    import time

    os.environ["PARAM_DECOMP_OUT_DIR"] = "/data/param_decomp_out"

    import numpy as np
    import torch

    import param_decomp.pretrain.run_info as pri

    def _local_files(entity, project, run_id):
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
    sites, sel = _site_groups(clusters)
    print(f"old: {len(clusters)} clusters over {len(sites)} sites, "
          f"{sum(len(v) for v in sel.values())} comps", flush=True)

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to(device)
    sel_t = {s: torch.as_tensor(v, device=device) for s, v in sel.items()}

    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")])
    B = 16
    pos_counts = {s: np.zeros((len(sel[s]), 512), np.int64) for s in sites}
    eos_counts = {s: np.zeros(len(sel[s]), np.int64) for s in sites}
    tok_counts = {s: [np.zeros(VOCAB, np.int64) for _ in cls]
                  for s, cls in sites.items()}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), B):
            tok = rows[i:i + B].to(device)
            out = model(tok, cache_type="input")
            ci = model.calc_causal_importances(out.cache, sampling="continuous")
            fires_np = {s: (ci.lower_leaky[s][..., sel_t[s]] > FIRE_THRESH)
                        .cpu().numpy() for s in sites}
            _accumulate(sites, sel, fires_np, rows[i:i + B].numpy(),
                        pos_counts, eos_counts, tok_counts)
            if (i // B) % 50 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    _save("old", sites, sel, pos_counts, eos_counts, tok_counts)
    print(f"old done ({time.time() - t0:.0f}s total)")


@app.function(image=jax_image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute_new(name: str, clusters: list[dict]) -> None:
    import time

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
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    sites, sel = _site_groups(clusters)
    sel_j = {s: jnp.asarray(v) for s, v in sel.items()}
    print(f"{name}: {len(clusters)} clusters over {len(sites)} sites, "
          f"{sum(len(v) for v in sel.values())} comps", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {s: jnp.clip(pre[s][..., sel_j[s]], 0.0, 1.0) > FIRE_THRESH
                for s in sites}

    rows = np.load("/data/pile_rows.npy")
    B = 16
    pos_counts = {s: np.zeros((len(sel[s]), rows.shape[1]), np.int64) for s in sites}
    eos_counts = {s: np.zeros(len(sel[s]), np.int64) for s in sites}
    tok_counts = {s: [np.zeros(VOCAB, np.int64) for _ in cls]
                  for s, cls in sites.items()}
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            tok = rows[i:i + B]
            out = step(placed, ci_fn, jnp.asarray(tok))
            fires_np = {s: np.asarray(out[s]) for s in sites}
            _accumulate(sites, sel, fires_np, tok,
                        pos_counts, eos_counts, tok_counts)
            if (i // B) % 50 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    _save(name, sites, sel, pos_counts, eos_counts, tok_counts)
    print(f"{name} done ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    path = Path(__file__).parent / "cache" / "l0_attn_clusters.json"
    all_clusters = json.loads(path.read_text())
    calls = [compute_old.spawn(all_clusters["old"]),
             compute_new.spawn("newA", all_clusters["newA"]),
             compute_new.spawn("newB", all_clusters["newB"])]
    for c in calls:
        c.get()
