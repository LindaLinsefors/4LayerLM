"""Modal GPU job: per-matrix CI co-activation (co-CI) correlation matrices for
ALL components of ALL matrices, both models.

For each module m and every pair of components (a, b) in it, accumulates the
sufficient statistics of Pearson r over per-token causal importance
(lower_leaky, sampling="continuous"): S1 = sum(c), G = sum(c cᵀ), plus
F = #tokens with CI > 0.1 per component. r is computed on-GPU in float64 and
saved as float16 (plenty for a heatmap); components that never vary get NaN.

Data (volume "vpd-4layer"):
  pile_4l   /vpd/model_400000.pth,        /pile_rows.pt      (4000 rows x 512)
  simple_2l /vpd-simple/model_400000.pth, /simple_stories.pt (6000 stories,
            variable length, truncated to 512; batches padded, pad positions
            masked out of all statistics)
  /param_decomp_out/pretrain_cache/spd-{t-9d2b8f02,gf6rbga0}/ -- target caches
  (WandB bypassed by monkeypatching the pretrain-cache downloader.)

Run:  modal run coci-heatmaps/hide/coci_compute_modal.py
Then: modal volume get vpd-4layer /coci_<model>.npz coci-heatmaps/hide/cache/
Output npz keys per module: "<mod>|mean" (C, float64), "<mod>|F" (C, float64),
"<mod>|r" (C x C, float16); plus "T" (token count).
"""

import modal

app = modal.App("coci-heatmaps")
vol = modal.Volume.from_name("vpd-4layer")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)

N_LAYERS = {"pile_4l": 4, "simple_2l": 2}
CKPT = {"pile_4l": "/data/vpd/model_400000.pth",
        "simple_2l": "/data/vpd-simple/model_400000.pth"}


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(model_name: str) -> None:
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
    FIRE_THRESH = 0.1

    model = ComponentModel.from_pretrained(CKPT[model_name])
    model.eval().to(device)
    modules = [f"h.{l}.{m}" for l in range(N_LAYERS[model_name])
               for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                         "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]
    C = {m: dict(model.named_parameters())
         [f"_components.{m.replace('.', '-')}.U"].shape[0] for m in modules}

    # batches: list of (tokens (B, S), mask (B, S) or None)
    if model_name == "pile_4l":
        rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")])
        batches = [(rows[i:i + 16], None) for i in range(0, len(rows), 16)]
    else:
        stories = [s[:512] for s in torch.load("/data/simple_stories.pt")]
        stories.sort(key=len)  # minimize padding within a batch
        batches = []
        for i in range(0, len(stories), 64):
            chunk = stories[i:i + 64]
            S = max(len(s) for s in chunk)
            tok = torch.zeros(len(chunk), S, dtype=torch.long)
            mask = torch.zeros(len(chunk), S, dtype=torch.bool)
            for j, s in enumerate(chunk):
                tok[j, :len(s)] = s
                mask[j, :len(s)] = True
            batches.append((tok, mask))

    S1 = {m: torch.zeros(C[m], device=device, dtype=torch.float64) for m in modules}
    F = {m: torch.zeros(C[m], device=device, dtype=torch.float64) for m in modules}
    G = {m: torch.zeros(C[m], C[m], device=device, dtype=torch.float64)
         for m in modules}
    T = 0

    t0 = time.time()
    with torch.no_grad():
        for b, (tok, mask) in enumerate(batches):
            tok = tok.to(device)
            out = model(tok, cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            for m in modules:
                c = ci_out.lower_leaky[m]                       # (B, S, C)
                c = (c.reshape(-1, C[m]) if mask is None
                     else c[mask.to(device)]).float()
                S1[m] += c.sum(0).double()
                F[m] += (c > FIRE_THRESH).sum(0).double()
                G[m] += (c.T @ c).double()
            T += tok.numel() if mask is None else int(mask.sum())
            if b % 25 == 0:
                print(f"batch {b + 1}/{len(batches)}, {T} tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)

    data = {"T": np.array(T)}
    for m in modules:
        mean = S1[m] / T
        cov = G[m] / T - torch.outer(mean, mean)
        s = torch.sqrt(torch.clamp(torch.diag(cov), min=0))
        r = cov / torch.outer(s, s)                             # 0-var -> inf/nan
        r[~torch.isfinite(r)] = torch.nan
        data[f"{m}|mean"] = mean.cpu().numpy()
        data[f"{m}|F"] = F[m].cpu().numpy()
        data[f"{m}|r"] = r.cpu().numpy().astype(np.float16)
    np.savez(f"/data/coci_{model_name}.npz", **data)
    vol.commit()
    print(f"saved /data/coci_{model_name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    list(compute.map(["pile_4l", "simple_2l"]))
