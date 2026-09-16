"""Modal GPU version of groups_compute.py: co-activation / co-CI sufficient
statistics for all alive q/k components, over 4000 Pile rows (2.05M tokens).

Volume "vpd-4layer" layout (uploaded beforehand):
  /vpd/model_400000.pth + final_config.yaml     -- VPD decomposition checkpoint
  /param_decomp_out/pretrain_cache/spd-t-9d2b8f02/  -- target-model cache
  /pile_rows.pt, /mean_ci.npz

WandB is bypassed by monkeypatching the pretrain-cache downloader to the
uploaded files, so no credentials are needed.

Run:  modal run pile-qk-comps/hide/groups_compute_modal.py
Then: modal volume get vpd-4layer /groups_stats.npz pile-qk-comps/hide/cache/groups_stats.npz
"""

import modal

app = modal.App("qk-groups-stats")
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

    acc = {m: {k: torch.zeros(len(alive[m]), device=device, dtype=torch.float64)
               for k in ("S1a", "S1c", "F")}
           | {k: torch.zeros(len(alive[m]), len(alive[m]), device=device,
                             dtype=torch.float64)
              for k in ("Ga", "Gc", "FF")}
           for m in MATRICES}

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N_ROWS, BATCH):
            out = model(rows[i:i + BATCH], cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            for m in MATRICES:
                idx = alive[m].to(device)
                a = ((out.cache[m] @ V[m][:, idx]) * u_norm[m][idx])
                a = a.reshape(-1, len(idx)).abs().double()
                c = ci_out.lower_leaky[m][:, :, idx].reshape(-1, len(idx)).double()
                f = (c > FIRE_THRESH).double()
                acc[m]["S1a"] += a.sum(0)
                acc[m]["Ga"] += a.T @ a
                acc[m]["S1c"] += c.sum(0)
                acc[m]["Gc"] += c.T @ c
                acc[m]["F"] += f.sum(0)
                acc[m]["FF"] += f.T @ f
            if (i // BATCH) % 25 == 0:
                print(f"{i + BATCH}/{N_ROWS} rows, {time.time() - t0:.0f}s", flush=True)

    data = {"T": np.array(N_ROWS * 512)}
    for m in MATRICES:
        for k, v in acc[m].items():
            data[f"{m}|{k}"] = v.cpu().numpy()
        data[f"{m}|idx"] = alive[m].numpy()
    np.savez("/data/groups_stats.npz", **data)
    vol.commit()
    print(f"saved /data/groups_stats.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    compute.remote()
