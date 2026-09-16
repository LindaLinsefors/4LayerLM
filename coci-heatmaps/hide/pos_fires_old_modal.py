"""Modal GPU job: per-position CI fire counts for the OLD (paper) decomposition
s-55ea3f9b — the pos_fires_new_modal.py analogue via the Torch ComponentModel
path (same forward/CI pipeline as coci_compute_modal.py, pile_4l only).

For every component of every matrix: how often it fires (CI > 0.1) at each
chunk position, over the 4,000 cached Pile rows. Used to flag pos-0
(attention-sink) components in the interactive cross heatmaps' hovers/ticks.

Run:  modal run coci-heatmaps/hide/pos_fires_old_modal.py
Then: modal volume get vpd-4layer /pos_fires_old.npz coci-heatmaps/hide/cache/
Output npz keys per module: "<mod>|Fp" (C, 512) int32 fire counts per position.
"""

import modal

app = modal.App("pos-fires-old-decomp")
vol = modal.Volume.from_name("vpd-4layer")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)

FIRE_THRESH = 0.1


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

    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to(device)
    modules = [f"h.{l}.{m}" for l in range(4)
               for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                         "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]
    C = {m: dict(model.named_parameters())
         [f"_components.{m.replace('.', '-')}.U"].shape[0] for m in modules}

    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")])
    acc = {m: torch.zeros(512, C[m], device=device, dtype=torch.int64)
           for m in modules}

    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(rows), 16):
            tok = rows[b:b + 16].to(device)
            out = model(tok, cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            for m in modules:
                acc[m] += (ci_out.lower_leaky[m] > FIRE_THRESH).sum(0)  # (S, C)
            if (b // 16) % 25 == 0:
                print(f"batch {b // 16 + 1}/{len(rows) // 16}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    data = {f"{m}|Fp": acc[m].T.cpu().numpy().astype(np.int32) for m in modules}
    np.savez_compressed("/data/pos_fires_old.npz", **data)
    vol.commit()
    print(f"saved /data/pos_fires_old.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    compute.remote()
