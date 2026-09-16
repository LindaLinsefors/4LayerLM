"""Modal GPU job: per-chunk-position CI statistics for all alive q/k
components, over 4000 Pile rows (2.05M tokens; same data as groups_stats).

For each matrix and alive component c, accumulates over the 4000 rows:
  Sp[c, p] = sum of CI (lower_leaky) at chunk position p (p = 0..511)
  Fp[c, p] = count of CI > 0.1 at position p
  Ap[c, p] = sum of |a_c| = |(V_c . input)| * ||U_c||  (activation magnitude)
  Fa[c, p] = count of |a_c| > 1
Each position is seen exactly 4000 times, so a position-independent component
has a flat profile up to sampling noise.

Volume "vpd-4layer" layout: see groups_compute_modal.py (same inputs).

Run:  modal run pile-qk-comps/hide/pos_ci_compute_modal.py
Then: modal volume get vpd-4layer /pos_ci.npz pile-qk-comps/hide/cache/pos_ci.npz
"""

import modal

app = modal.App("qk-pos-ci")
vol = modal.Volume.from_name("vpd-4layer")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute() -> None:
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

    N_ROWS, BATCH, FIRE_THRESH = 4000, 16, 0.1
    MATRICES = [f"h.{l}.attn.{m}" for l in range(4) for m in ("q_proj", "k_proj")]

    device = "cuda"
    mean_ci = np.load("/data/mean_ci.npz")
    alive = {m: torch.tensor(np.where(mean_ci[m] > 1e-6)[0]) for m in MATRICES}
    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")[:N_ROWS]])

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to(device)
    params = dict(model.named_parameters())
    V = {m: params[f"_components.{m.replace('.', '-')}.V"].detach() for m in MATRICES}
    u_norm = {m: params[f"_components.{m.replace('.', '-')}.U"].detach().norm(dim=1)
              for m in MATRICES}

    acc = {m: {k: torch.zeros(len(alive[m]), 512, device=device,
                              dtype=torch.float64)
               for k in ("Sp", "Fp", "Ap", "Fa")}
           for m in MATRICES}

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N_ROWS, BATCH):
            out = model(rows[i:i + BATCH], cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            for m in MATRICES:
                idx = alive[m].to(device)
                c = ci_out.lower_leaky[m][:, :, idx].double()   # (B, 512, C)
                acc[m]["Sp"] += c.sum(0).T
                acc[m]["Fp"] += (c > FIRE_THRESH).double().sum(0).T
                a = ((out.cache[m] @ V[m][:, idx]) * u_norm[m][idx]).abs().double()
                acc[m]["Ap"] += a.sum(0).T
                acc[m]["Fa"] += (a > 1).double().sum(0).T
            if (i // BATCH) % 25 == 0:
                print(f"{i + BATCH}/{N_ROWS} rows, {time.time() - t0:.0f}s", flush=True)

    data = {"n_rows": np.array(N_ROWS)}
    for m in MATRICES:
        for k, v in acc[m].items():
            data[f"{m}|{k}"] = v.cpu().numpy()
        data[f"{m}|idx"] = alive[m].numpy()
    np.savez("/data/pos_ci.npz", **data)
    vol.commit()
    print(f"saved /data/pos_ci.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    compute.remote()
