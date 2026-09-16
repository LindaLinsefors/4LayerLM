"""Modal GPU job: CI statistics on hyphen tokens for ALL components of all 24
decomposed matrices, over the 4000 cached Pile rows (2.048M tokens).

Position sets from find_hyphens.py (uploaded to the volume as
hn_positions.npz): hyphen = every token 428 (' -'); cand = the 17 auto-detected
heading candidates (full per-position CI saved so borderline calls can be
checked); hn = the 12 manually confirmed hacker-news title|username hyphens
(a subset of cand — their stats are derived locally from CI_cand).

Per matrix m and component c, accumulated over all rows:
  m|S_all[c]   = sum of CI (lower_leaky) over all positions
  m|F_all[c]   = count of CI > 0.1 over all positions
  m|S_hyph[c]  = sum of CI over hyphen positions
  m|F_hyph[c]  = count of CI > 0.1 over hyphen positions
  m|CI_cand[c, k] = CI at candidate position k (order of cand_rc)

Volume "vpd-4layer" layout: see pile-qk-comps/hide/groups_compute_modal.py.
First: modal volume put vpd-4layer hacker-news-hyfen/hide/cache/hn_positions.npz hn_positions.npz
Run (set PYTHONUTF8=1 PYTHONIOENCODING=utf-8 on Windows):
  modal run hacker-news-hyfen/hide/ci_compute_modal.py
Then: cd hacker-news-hyfen/hide/cache && modal volume get vpd-4layer hn_ci.npz
"""

import modal

app = modal.App("hn-hyphen-ci")
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

    device = "cuda"
    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")[:N_ROWS]])
    pos = np.load("/data/hn_positions.npz")
    hyph_mask = torch.zeros(N_ROWS, 512, dtype=torch.bool)
    hyph_mask[pos["hyphen_rc"][:, 0], pos["hyphen_rc"][:, 1]] = True
    cand_rc = pos["cand_rc"]  # (17, 2), row-sorted

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to(device)

    acc: dict[str, dict[str, torch.Tensor]] = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N_ROWS, BATCH):
            out = model(rows[i:i + BATCH].to(device), cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            hm = hyph_mask[i:i + BATCH].to(device)
            cand_here = [(k, r - i, p) for k, (r, p) in enumerate(cand_rc)
                         if i <= r < i + BATCH]
            for m, c in ci_out.lower_leaky.items():
                c = c.double()                                # (B, 512, C)
                if m not in acc:
                    C = c.shape[-1]
                    z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)
                    acc[m] = {"S_all": z(C), "F_all": z(C),
                              "S_hyph": z(C), "F_hyph": z(C),
                              "CI_cand": z(C, len(cand_rc))}
                acc[m]["S_all"] += c.sum((0, 1))
                acc[m]["F_all"] += (c > FIRE_THRESH).double().sum((0, 1))
                acc[m]["S_hyph"] += c[hm].sum(0)
                acc[m]["F_hyph"] += (c[hm] > FIRE_THRESH).double().sum(0)
                for k, b, p in cand_here:
                    acc[m]["CI_cand"][:, k] = c[b, p]
            if (i // BATCH) % 25 == 0:
                print(f"{i + BATCH}/{N_ROWS} rows, {time.time() - t0:.0f}s", flush=True)

    data = {"n_rows": np.array(N_ROWS), "n_pos": np.array(N_ROWS * 512),
            "n_hyph": np.array(int(hyph_mask.sum())), "cand_rc": cand_rc}
    for m in acc:
        for k, v in acc[m].items():
            data[f"{m}|{k}"] = v.cpu().numpy()
    np.savez("/data/hn_ci.npz", **data)
    vol.commit()
    print(f"saved /data/hn_ci.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    compute.remote()
