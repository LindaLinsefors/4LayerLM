"""Test GPT-OSS-style attention conventions on the sink checkpoints.

The fitted per-plane frequencies (fit_rope_freqs.py) match rope theta 150000 on
fast planes with extra slowdown on slow planes — the YaRN-by-parts signature of
GPT-OSS (theta 150000, factor 32, original ctx 4096, beta 32/1, attention
scaling 0.1*ln(32)+1). GPT-OSS also uses sliding-window-128 attention on
alternating layers. This script scores those combinations.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_sink

BANDS = [(0, 32), (32, 128), (128, 256), (256, 511)]


def yarn_cos_sin(T, theta=150000.0, factor=32.0, orig_ctx=4096,
                 beta_fast=32.0, beta_slow=1.0, mscale=True):
    """GPT-OSS / HF YaRN ("ntk-by-parts") cos/sin, rotate-half layout."""
    p = torch.arange(64, dtype=torch.float64)
    inv_freq = theta ** (-p / 64)
    # number of rotations each plane completes in the original context
    rotations = orig_ctx * inv_freq / (2 * math.pi)
    ramp = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)
    inv_freq = inv_freq * ((1 - ramp) / factor + ramp)
    scale = (0.1 * math.log(factor) + 1.0) if mscale else 1.0
    pos = torch.arange(T, dtype=torch.float64)
    ang = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return (emb.cos() * scale).float(), (emb.sin() * scale).float()


def plain_cos_sin(T, base):
    p = torch.arange(64, dtype=torch.float64)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * (base ** (-p / 64))[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos().float(), emb.sin().float()


def make_forward(cos, sin, window=None):
    def fwd(self, x):
        B, T, C = x.size()
        hd = self.head_dim
        q = self.q_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)

        def rot(t):
            n = t.shape[-1] // 2
            return torch.cat([-t[..., n:], t[..., :n]], dim=-1)

        c, s = cos[:T], sin[:T]
        q = q * c + rot(q) * s
        k = k * c + rot(k) * s
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        if window is not None:
            mask &= ~torch.ones(T, T, dtype=torch.bool).tril(diagonal=-window)
        att = att.masked_fill(~mask, float("-inf"))
        sink = self.sinks.view(1, -1, 1, 1).expand(B, -1, T, 1)
        att = F.softmax(torch.cat([att, sink], dim=-1), dim=-1)
        return self.o_proj(
            (att[..., :-1] @ v).transpose(1, 2).contiguous().view(B, T, C))
    return fwd


def score(model, batch, cos, sin, windows):
    """windows: per-layer sliding window (None = full)."""
    for block, w in zip(model.h, windows):
        block.attn.forward = types.MethodType(make_forward(cos, sin, w), block.attn)
    with torch.no_grad():
        lo = model(batch)
    logp = lo.log_softmax(-1)
    nll = -logp[:, :-1].gather(-1, batch[:, 1:, None])[..., 0].numpy()
    return [nll[:, a:b].mean() for a, b in BANDS], nll.mean()


def main():
    model, _ = load_sink(45)
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)[:8]
    batch = torch.stack([r[:512] for r in rows])
    T = 512

    full = [None] * 4
    alt_even = [128, None, 128, None]   # GPT-OSS order: sliding first
    alt_odd = [None, 128, None, 128]

    variants = [
        ("plain 1e6, full (ref)", plain_cos_sin(T, 1e6), full),
        ("yarn gptoss, full", yarn_cos_sin(T), full),
        ("yarn gptoss no-mscale, full", yarn_cos_sin(T, mscale=False), full),
        ("yarn gptoss, window even layers", yarn_cos_sin(T), alt_even),
        ("yarn gptoss, window odd layers", yarn_cos_sin(T), alt_odd),
        ("plain theta 150000, full", plain_cos_sin(T, 150000.0), full),
        ("yarn theta 1e4 f32, full", yarn_cos_sin(T, theta=1e4), full),
        ("yarn gptoss orig_ctx 512, full", yarn_cos_sin(T, orig_ctx=512), full),
    ]
    print(f"{'variant':36s}  " + "  ".join(f"{a}-{b}" for a, b in BANDS) + "  overall")
    for name, (cos, sin), win in variants:
        bands, tot = score(model, batch, cos, sin, win)
        print(f"{name:36s}  " + "  ".join(f"{v:6.3f}" for v in bands) + f"  {tot:.3f}")


if __name__ == "__main__":
    main()
