"""Modal GPU job: per-chunk-position CI statistics for ALL components of the
non-q/k matrices of layers 0-1 (v_proj, o_proj, mlp.c_fc, mlp.down_proj) —
the stages of the position-0 massive-vector mechanism. Complements
pile-qk-comps' pos_ci.npz (q/k of all layers, alive only).

For each matrix and component c, over 4000 Pile rows:
  Sp[c, p] = sum of CI (lower_leaky) at chunk position p (0..511)
  Fp[c, p] = count of CI > 0.1 at position p
  Se[c]    = sum of CI at mid-sequence <|endoftext|> positions
  Fe[c]    = count of CI > 0.1 at those positions
(EOS positions are also inside Sp/Fp at their scattered chunk positions; Se/Fe
isolate them. n_eos saved for normalization. Every chunk position is seen
exactly 4000 times; chunk cuts are random within documents, so flat = no
position dependence.)

Volume "vpd-4layer" layout: see pile-qk-comps/hide/groups_compute_modal.py.

Run (from project root; set PYTHONUTF8=1 PYTHONIOENCODING=utf-8 on Windows):
  modal run endoftext-pos0/hide/pos0_ci_compute_modal.py
Then:
  cd endoftext-pos0/hide/cache && modal volume get vpd-4layer pos0_ci.npz
"""

import modal

app = modal.App("pos0-ci")
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

    N_ROWS, BATCH, FIRE_THRESH, EOS_ID = 4000, 16, 0.1, 0
    MATRICES = [f"h.{l}.{m}" for l in (0, 1)
                for m in ("attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]

    device = "cuda"
    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")[:N_ROWS]])

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to(device)

    acc: dict[str, dict[str, torch.Tensor]] = {}
    n_eos = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N_ROWS, BATCH):
            batch = rows[i:i + BATCH].to(device)
            out = model(batch, cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            eos = (batch == EOS_ID)
            eos[:, 0] = False
            n_eos += int(eos.sum())
            for m in MATRICES:
                c = ci_out.lower_leaky[m].double()            # (B, 512, C)
                if m not in acc:
                    C = c.shape[-1]
                    acc[m] = {"Sp": torch.zeros(C, 512, device=device, dtype=torch.float64),
                              "Fp": torch.zeros(C, 512, device=device, dtype=torch.float64),
                              "Se": torch.zeros(C, device=device, dtype=torch.float64),
                              "Fe": torch.zeros(C, device=device, dtype=torch.float64)}
                acc[m]["Sp"] += c.sum(0).T
                acc[m]["Fp"] += (c > FIRE_THRESH).double().sum(0).T
                acc[m]["Se"] += c[eos].sum(0)
                acc[m]["Fe"] += (c[eos] > FIRE_THRESH).double().sum(0)
            if (i // BATCH) % 25 == 0:
                print(f"{i + BATCH}/{N_ROWS} rows, {time.time() - t0:.0f}s", flush=True)

    data = {"n_rows": np.array(N_ROWS), "n_eos": np.array(n_eos)}
    for m in MATRICES:
        for k, v in acc[m].items():
            data[f"{m}|{k}"] = v.cpu().numpy().astype(np.float32)
    np.savez("/data/pos0_ci.npz", **data)
    vol.commit()
    print(f"saved /data/pos0_ci.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    compute.remote()
