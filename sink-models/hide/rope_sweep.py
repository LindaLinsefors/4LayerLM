"""Empirical sweep over attention-convention variants for the sink models.

The shared sink checkpoints (t-87f91319 / t-75f6c439) reached val_loss 2.65 in
training (WandB), but both the official PR-#1002 JAX loader and our Torch port
(which agree to 3 decimals) give NLL ~8.1 on real Pile rows — good at short
range, confidently wrong at long range. The emitted model_config.yaml states
the rotary convention as a port constant rather than recording it from the
training run, so this script sweeps plausible training-time conventions and
reports NLL by position band. The right convention should hit ~2.6 everywhere.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_sink

N_ROWS = 8


def patched_forward(self, x, *, pairing="half", rotary_dim=128, base=10000.0,
                    use_rope=True, window=None):
    """CausalSelfAttention.forward with configurable RoPE/window variants."""
    B, T, C = x.size()
    hd = self.head_dim
    q = self.q_proj(x).view(B, T, self.n_head, hd).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_key_value_heads, hd).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_key_value_heads, hd).transpose(1, 2)

    if use_rope:
        pos = torch.arange(T, dtype=torch.float32)
        dim = torch.arange(rotary_dim // 2, dtype=torch.float32)
        inv_freq = base ** (-dim / (rotary_dim // 2))
        angles = pos[:, None] * inv_freq[None, :]  # (T, rd/2)
        if pairing == "half":
            emb = torch.cat([angles, angles], dim=-1)  # (T, rd)
            cos, sin = emb.cos(), emb.sin()

            def rot(t):
                n = t.shape[-1] // 2
                return torch.cat([-t[..., n:], t[..., :n]], dim=-1)
        else:  # adjacent pairs (interleaved, GPT-NeoX rotate_every_two)
            emb = angles.repeat_interleave(2, dim=-1)  # (T, rd)
            cos, sin = emb.cos(), emb.sin()

            def rot(t):
                out = torch.empty_like(t)
                out[..., ::2] = -t[..., 1::2]
                out[..., 1::2] = t[..., ::2]
                return out

        def apply(t):
            r, rest = t[..., :rotary_dim], t[..., rotary_dim:]
            r = r * cos + rot(r) * sin
            return torch.cat([r, rest], dim=-1)

        q, k = apply(q), apply(k)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    if window is not None:
        mask &= ~torch.ones(T, T, dtype=torch.bool).tril(diagonal=-window)
    att = att.masked_fill(~mask, float("-inf"))
    sink = self.sinks.view(1, -1, 1, 1).expand(B, -1, T, 1)
    att = F.softmax(torch.cat([att, sink.to(att.dtype)], dim=-1), dim=-1)
    y = (att[..., :-1] @ v).transpose(1, 2).contiguous().view(B, T, C)
    return self.o_proj(y)


def run(model, batch, **kw):
    import types

    for block in model.h:
        block.attn.forward = types.MethodType(
            lambda self, x, kw=kw: patched_forward(self, x, **kw), block.attn)
    with torch.no_grad():
        lo = model(batch)
    logp = lo.log_softmax(-1)
    nll = -logp[:, :-1].gather(-1, batch[:, 1:, None])[..., 0].numpy()
    return {b: nll[:, a:b].mean() for a, b in
            [(0, 32), (32, 128), (128, 256), (256, 511)]}


def main():
    model, _ = load_sink(45)
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)[:N_ROWS]
    batch = torch.stack([r[:512] for r in rows])

    variants = [
        ("baseline (rotate-half, rd=128, base 1e4)", {}),
        ("adjacent pairs", dict(pairing="adjacent")),
        ("no rope", dict(use_rope=False)),
        ("partial rotary rd=64", dict(rotary_dim=64)),
        ("partial rotary rd=32", dict(rotary_dim=32)),
        ("base 1e5", dict(base=1e5)),
        ("base 1e6", dict(base=1e6)),
        ("sliding window 64", dict(window=64)),
        ("sliding window 128", dict(window=128)),
        ("sliding window 256", dict(window=256)),
        ("adjacent + window 128", dict(pairing="adjacent", window=128)),
    ]
    print(f"{'variant':38s}  pos<32   32-128  128-256  256-511")
    for name, kw in variants:
        r = run(model, batch, **kw)
        print(f"{name:38s}  " + "  ".join(f"{v:6.3f}" for v in r.values()))


if __name__ == "__main__":
    main()
