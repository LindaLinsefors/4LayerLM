"""Modal GPU job: where in sink seed 45's residual stream does the PREVIOUS
token become linearly readable?

At every stage of the residual stream (after embedding, after each attention,
after each MLP, after ln_f) train a probe to predict the token at position i-1
from the stream at position i, in two variants:

  linear      logits = x @ W          (a trained unembedding, one read-out
                                       vector per vocab token; no bias, like
                                       the model's own lm_head)
  norm+linear logits = (g * x/rms(x)) @ W   (the model's RMSNorm — learnable
                                       gain g, eps 1e-6 — trained jointly)

Model: sink seed 45 (t-87f91319) with the fitted corrected RoPE spectrum
(sink-models/rope_report.md) installed into the Torch port's rotary buffers —
same as load.load_sink(45); an NLL sanity assert in-job guards the install.

Data: the first 2,000 of the 4,000 cached Pile rows (/pile_rows.npy on the
volume), split by row into 1,600 train / 400 test. Positions 1..511 (position 0
has no previous token); 817,600 train / 204,400 test samples. All 20 probes
(10 stages x 2 variants) run as parallel Modal containers (A10G); each
recomputes the cheap forward capture (~10 s) for just its stage, then trains:
Adam, batch 8,192, 12 epochs cosine lr 3e-3 -> 0 (test top-1 also recorded at
the halfway point as a convergence check).

Run:  modal run prev-token/hide/prev_probe_modal.py     (~10 min wall)
Results are returned over the wire and written to hide/cache/prev_probe.npz
(also saved on the volume as /prev_probe_sink45.npz in case the client dies).
"""

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("prev-token-probe")
vol = modal.Volume.from_name("vpd-4layer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "safetensors", "pyyaml")
)
if modal.is_local():  # local paths don't exist when the container imports this file
    ROOT = HERE.parents[1]
    image = (
        image
        .add_local_file(ROOT / "prev_paper" / "model_def.py", "/root/model_def.py")
        .add_local_file(ROOT / "sink-models" / "hide" / "cache" / "fitted_freqs_avg.npz",
                        "/root/fitted_freqs_avg.npz")
    )

N_ROWS, N_TRAIN = 2000, 1600  # rows; test = the rest
BATCH, EPOCHS, LR = 8192, 12, 3e-3
STAGES = (["emb"]
          + [f"{kind}{l + 1}" for l in range(4) for kind in ("attn", "mlp")]
          + ["ln_f"])


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def probe_one(stage: str, with_norm: bool) -> dict:
    import math
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from safetensors.torch import load_file

    sys.path.insert(0, "/root")
    from model_def import LlamaSimpleMLP, LlamaSimpleMLPConfig

    dev = "cuda"
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")  # TF32: ~2x faster probe matmuls

    # --- model: sink seed 45 with the fitted corrected RoPE ---
    base = Path("/data/sink-models/pretrain_cache/spd-t-87f91319")
    cfg = yaml.safe_load((base / "model_config.yaml").read_text())
    cfg.setdefault("flash_attention", False)
    model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**cfg))
    model.load_state_dict(load_file(base / "model_step_100000.safetensors"))
    inv_freq = torch.tensor(np.exp(np.load("/root/fitted_freqs_avg.npz")["log_inv_freq"]),
                            dtype=torch.float32)
    angles = torch.arange(cfg["n_ctx"], dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([angles, angles], dim=-1)
    for block in model.h:
        block.attn.rotary_sin.copy_(emb.sin())
        block.attn.rotary_cos.copy_(emb.cos())
    model.eval().requires_grad_(False).to(dev)

    rows = torch.tensor(np.load("/data/pile_rows.npy")[:N_ROWS].astype(np.int64))

    # RoPE sanity: fitted spectrum ~2.9 NLL on Pile rows, broken ~8
    logits = model(rows[:8].to(dev))
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                          rows[:8, 1:].reshape(-1).to(dev)).item()
    assert nll < 4.0, f"RoPE install broken? NLL {nll}"

    # --- capture the residual stream at this probe's stage (fp16 on GPU) ---
    d, T = cfg["n_embd"], rows.shape[1]
    X = torch.empty(N_ROWS, T, d, dtype=torch.float16, device=dev)
    for i in range(0, N_ROWS, 32):
        idx = rows[i:i + 32].to(dev)
        with torch.no_grad():
            x = model.wte(idx)
            if stage == "emb":
                X[i:i + 32] = x.half()
                continue
            for l, block in enumerate(model.h):
                x = x + block.attn(block.rms_1(x))
                if stage == f"attn{l + 1}":
                    break
                x = x + block.mlp(block.rms_2(x))
                if stage == f"mlp{l + 1}":
                    break
            else:
                x = model.ln_f(x)  # stage == "ln_f"
            X[i:i + 32] = x.half()
    del model
    torch.cuda.empty_cache()

    # targets: previous token; sample (row, pos) uses stream at pos, predicts pos-1
    y_all = rows[:, :-1]                      # (N_ROWS, T-1)
    V, eps = cfg["vocab_size"], cfg["rms_norm_eps"]
    n_tr = N_TRAIN * (T - 1)

    def evaluate(W, g, Xe, y):
        """NLL, top-1, top-5 + per-position top-1 counts. Xe (n, T-1, d) fp16 on GPU."""
        nll = correct = top5 = 0.0
        pos_correct = torch.zeros(T - 1, dtype=torch.float64)
        with torch.no_grad():
            for j in range(0, Xe.shape[0], 16):       # 16 rows x 511 pos per batch
                xb = Xe[j:j + 16].float()
                if g is not None:
                    xb = g * (xb * torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + eps))
                lg = xb @ W                                    # (b, T-1, V)
                yb = y[j:j + 16].to(dev)
                nll += F.cross_entropy(lg.reshape(-1, V), yb.reshape(-1),
                                       reduction="sum").item()
                hit = lg.argmax(-1) == yb                      # (b, T-1)
                correct += hit.sum().item()
                top5 += (lg.topk(5, dim=-1).indices == yb[..., None]).any(-1).sum().item()
                pos_correct += hit.sum(0).double().cpu()
        n = Xe.shape[0] * (T - 1)
        return nll / n, correct / n, top5 / n, (pos_correct / Xe.shape[0]).numpy()

    # --- train ---
    Xtr = X[:N_TRAIN, 1:].reshape(-1, d)                   # stream at pos 1..511
    ytr = y_all[:N_TRAIN].reshape(-1).to(dev)
    W = torch.zeros(d, V, device=dev, requires_grad=True)
    params = [W]
    g = None
    if with_norm:
        g = torch.ones(d, device=dev, requires_grad=True)
        params.append(g)
    opt = torch.optim.Adam(params, lr=LR)
    steps_total = EPOCHS * (n_tr // BATCH)
    step = 0
    mid_top1 = None
    t1 = time.time()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n_tr, device=dev)
        for j in range(0, n_tr - BATCH + 1, BATCH):
            b = perm[j:j + BATCH]
            xb = Xtr[b].float()
            if with_norm:
                xb = g * (xb * torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + eps))
            loss = F.cross_entropy(xb @ W, ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # cosine decay to 0
            for pg in opt.param_groups:
                pg["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / steps_total))
            opt.step()
            step += 1
        if epoch == EPOCHS // 2 - 1:
            _, mid_top1, _, _ = evaluate(W, g, X[N_TRAIN:, 1:], y_all[N_TRAIN:])
    te_nll, te_top1, te_top5, pos_acc = evaluate(W, g, X[N_TRAIN:, 1:], y_all[N_TRAIN:])
    tr_nll, tr_top1, _, _ = evaluate(W, g, X[:200, 1:], y_all[:200])
    out = dict(test_nll=te_nll, test_top1=te_top1, test_top5=te_top5,
               mid_top1=mid_top1, train_nll=tr_nll, train_top1=tr_top1,
               pos_acc=pos_acc.tolist())
    if with_norm:
        out["gain"] = g.detach().cpu().numpy().tolist()
    print(f"{stage:6s} {'norm+lin' if with_norm else 'linear  '}: "
          f"test top-1 {te_top1:.4f} (mid {mid_top1:.4f})  top-5 {te_top5:.4f}  "
          f"NLL {te_nll:.3f}  [train top-1 {tr_top1:.4f}]  "
          f"{time.time() - t1:.0f}s", flush=True)
    return out


@app.function(image=image, volumes={"/data": vol}, timeout=7200)
def save_backup(results: dict) -> None:
    import numpy as np
    np.savez_compressed("/data/prev_probe_sink45.npz",
                        **{k: np.asarray(v) for k, v in results.items()})
    vol.commit()


@app.local_entrypoint()
def main() -> None:
    import numpy as np

    jobs = [(s, wn) for s in STAGES for wn in (False, True)]
    results = {}
    for (stage, wn), out in zip(jobs, probe_one.starmap(jobs)):
        key = f"{stage}|{'norm' if wn else 'lin'}"
        for name, val in out.items():
            if val is not None:
                results[f"{key}|{name}"] = val
    save_backup.remote(results)
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache / "prev_probe.npz",
                        **{k: np.asarray(v) for k, v in results.items()})
    print("saved", cache / "prev_probe.npz")
