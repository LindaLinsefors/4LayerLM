"""Modal GPU job: per-token-id CI sums for the top-20 sample-mean-CI components
of the OLD (paper) decomposition s-55ea3f9b, h.0.attn.q_proj AND h.0.attn.k_proj,
over the 4,000 cached Pile rows — the components/Old analogue of
ci_per_token_modal_C.py, via the Torch ComponentModel path
(loading template: coci-heatmaps/hide/pos_fires_old_modal.py). Both matrices are
accumulated in one forward pass. CI = lower_leaky, sampling="continuous" (the
co-CI convention); component ids selected locally from coci_pile_4l.npz sample
mean CI. No RoPE caveat: the old target loads correctly.

Output per matrix: cache/ci_per_token_old_<h0q|h0k>_top20.npz
("ids", "ci_sum", "fires") — summed CI and fire counts (CI > 0.1) per input
token id, (vocab, 20). Mean CI per token id = ci_sum / corpus count (local).

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run components/hide/ci_per_token_modal_old.py
"""

import modal

app = modal.App("old-components-ci-per-token")
vol = modal.Volume.from_name("vpd-4layer")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)

VOCAB = 50277
N_TOP = 20
FIRE_THRESH = 0.1


def short_name(mod: str) -> str:  # same rule as components/C
    p = mod.split(".")
    return f"h{p[1]}{p[3][0] if p[2] == 'attn' else p[3][0] + 'm'}"


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(comps_by_mod: dict) -> dict:
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

    idx = {m: torch.tensor(c, device=device) for m, c in comps_by_mod.items()}
    acc = {m: torch.zeros(VOCAB, len(c), device=device) for m, c in comps_by_mod.items()}
    fires = {m: torch.zeros(VOCAB, len(c), device=device, dtype=torch.int64)
             for m, c in comps_by_mod.items()}

    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")])
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(rows), 16):
            tok = rows[b:b + 16].to(device)
            out = model(tok, cache_type="input")
            ci_out = model.calc_causal_importances(out.cache, sampling="continuous")
            flat = tok.reshape(-1)
            for m in comps_by_mod:
                ci = ci_out.lower_leaky[m][..., idx[m]].reshape(len(flat), -1)
                acc[m].index_add_(0, flat, ci)
                fires[m].index_add_(0, flat, (ci > FIRE_THRESH).long())
            if (b // 16) % 50 == 0:
                print(f"batch {b // 16 + 1}/{len(rows) // 16}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    print(f"done, {time.time() - t0:.0f}s", flush=True)
    return {m: {"ci_sum": acc[m].cpu().numpy(),
                "fires": fires[m].cpu().numpy().astype(np.int32)}
            for m in comps_by_mod}


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    import numpy as np

    here = Path(__file__).resolve().parent
    coci = np.load(here.parents[1] / "coci-heatmaps" / "hide" / "cache"
                   / "coci_pile_4l.npz")
    comps_by_mod = {}
    for mod in ("h.0.attn.q_proj", "h.0.attn.k_proj"):
        mean = coci[f"{mod}|mean"]
        comps_by_mod[mod] = np.argsort(mean)[::-1][:N_TOP].tolist()
        print(f"top {N_TOP} of {mod}: {comps_by_mod[mod]}")

    out = compute.remote(comps_by_mod)
    (here / "cache").mkdir(exist_ok=True)
    for mod, d in out.items():
        path = here / "cache" / f"ci_per_token_old_{short_name(mod)}_top20.npz"
        np.savez_compressed(path, ids=np.asarray(comps_by_mod[mod], np.int32),
                            ci_sum=d["ci_sum"], fires=d["fires"])
        print("saved", path)
