"""Recover the sink models' training-time RoPE frequencies from the weights.

The checkpoints behave as if trained with a much slower RoPE than the recorded
rotary_base 10000 (see rope_sweep.py). Here we treat the 64 per-plane rotation
frequencies (shared across heads/layers, rotate-half convention) as free
parameters and minimize next-token NLL on cached Pile rows by Adam. If the
optimum is a geometric sequence base^(-p/64), the convention family is
confirmed and the base is read off directly.

Writes the fitted frequencies to cache/fitted_freqs.npz and prints the
per-plane implied base. Local GPU, ~2 min.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_sink

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_VAL = 48, 16
STEPS = 800


def forward_nll(model, batch, log_inv_freq):
    """Full forward with RoPE angles from log_inv_freq (64,); returns mean NLL."""
    B, T = batch.shape
    inv_freq = torch.exp(log_inv_freq)  # (64,)
    pos = torch.arange(T, device=batch.device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]  # (T, 64)
    emb = torch.cat([angles, angles], dim=-1)  # (T, 128)
    cos, sin = emb.cos(), emb.sin()

    def rot(t):
        n = t.shape[-1] // 2
        return torch.cat([-t[..., n:], t[..., :n]], dim=-1)

    x = model.wte(batch)
    for block in model.h:
        xn = block.rms_1(x)
        a = block.attn
        q = a.q_proj(xn).view(B, T, a.n_head, a.head_dim).transpose(1, 2)
        k = a.k_proj(xn).view(B, T, a.n_head, a.head_dim).transpose(1, 2)
        v = a.v_proj(xn).view(B, T, a.n_head, a.head_dim).transpose(1, 2)
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        att = (q @ k.transpose(-2, -1)) / a.head_dim**0.5
        mask = torch.ones(T, T, dtype=torch.bool, device=batch.device).tril()
        att = att.masked_fill(~mask, float("-inf"))
        sink = a.sinks.view(1, -1, 1, 1).expand(B, -1, T, 1)
        att = F.softmax(torch.cat([att, sink], dim=-1), dim=-1)
        y = (att[..., :-1] @ v).transpose(1, 2).reshape(B, T, -1)
        x = x + a.o_proj(y)
        x = x + block.mlp(block.rms_2(x))
    logits = model.lm_head(model.ln_f(x))
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    model, _ = load_sink(seed)
    model = model.to(DEVICE)
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)
    train = torch.stack([r[:512] for r in rows[:N_TRAIN]]).to(DEVICE)
    val = torch.stack([r[:512] for r in rows[N_TRAIN:N_TRAIN + N_VAL]]).to(DEVICE)

    # init: the best hand-found spectrum (yarn theta 150k, factor 4, orig ctx
    # 4096, no mscale — see gptoss_rope_test.py)
    sys.path.insert(0, str(HERE))
    from gptoss_rope_test import yarn_cos_sin  # noqa: E402
    import math
    p64 = torch.arange(64, dtype=torch.float64)
    inv0 = 150000.0 ** (-p64 / 64)
    rot_n = 4096 * inv0 / (2 * math.pi)
    ramp = ((rot_n - 1.0) / (32.0 - 1.0)).clamp(0, 1)
    inv0 = inv0 * ((1 - ramp) / 4.0 + ramp)
    log_inv_freq = torch.log(inv0).float().to(DEVICE).requires_grad_(True)

    p = torch.arange(64, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        print(f"val NLL @ config base 1e4: "
              f"{forward_nll(model, val, -(p / 64) * np.log(1e4)).item():.4f}")
        print(f"val NLL @ init (yarn f4): "
              f"{forward_nll(model, val, log_inv_freq).item():.4f}")

    opt = torch.optim.Adam([log_inv_freq], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for step in range(STEPS):
        idx = torch.randperm(N_TRAIN)[:8]
        loss = forward_nll(model, train[idx], log_inv_freq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                v = forward_nll(model, val, log_inv_freq).item()
            print(f"step {step + 1}: train {loss.item():.4f}  val {v:.4f}")

    lif = log_inv_freq.detach().cpu().numpy()
    implied_base = np.exp(-lif * 64 / np.maximum(np.arange(64), 1))  # per-plane
    np.savez(HERE / "cache" / f"fitted_freqs_{seed}.npz",
             log_inv_freq=lif, implied_base=implied_base)
    print("\nplane:  fitted inv_freq   implied base (if geometric)")
    for i in range(0, 64, 4):
        print(f"{i:3d}:  {np.exp(lif[i]):12.6g}   {implied_base[i]:12.6g}")
    print(f"saved cache/fitted_freqs_{seed}.npz")


if __name__ == "__main__":
    main()
