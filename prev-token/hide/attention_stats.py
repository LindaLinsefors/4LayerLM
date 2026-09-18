"""Per-head attention mass on the previous token (and on self / the sink slot),
for all four target models: pile_4l, sink seed 45, sink seed 46, simple_2l.

Pile models: 500 cached Pile rows (512 tokens each). simple_2l: 500 cached
stories (truncated to 512). Attention is recomputed per head from each
attention module's own weights via a forward pre-hook (the module itself may
run flash attention); for the sink models the softmax includes the learned
sink slot, so all fractions are shares of total attention including the sink.

Writes cache/attn_stats.npz and prints the report tables. ~3 min local GPU.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import load

dev = "cuda" if torch.cuda.is_available() else "cpu"

pile_rows = [r[:512] for r in torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")[:500]]
story_rows = torch.load(ROOT / "context-loss/hide/cache/simple_stories.pt")[:500]


def run(model, batches):
    model = model.to(dev).eval()
    L, H = len(model.h), model.h[0].attn.n_head
    prev_sum = torch.zeros(L, H, dtype=torch.float64)
    self_sum = torch.zeros(L, H, dtype=torch.float64)
    sink_sum = torch.zeros(L, H, dtype=torch.float64)
    n_q = 0

    def make_hook(l):
        def hook(mod, args):
            nonlocal n_q
            x = args[0]
            B, T, C = x.shape
            q = mod.q_proj(x).view(B, T, mod.n_head, mod.head_dim).transpose(1, 2)
            k = mod.k_proj(x).view(B, T, mod.n_key_value_heads, mod.head_dim).transpose(1, 2)
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            cos = mod.rotary_cos[pos].to(q.dtype); sin = mod.rotary_sin[pos].to(q.dtype)
            q, k = mod._apply_rotary_pos_emb(q, k, cos, sin)
            if mod.repeat_kv_heads > 1:
                k = k.repeat_interleave(mod.repeat_kv_heads, dim=1)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            att = att.masked_fill(mod.bias[:, :, :T, :T] == 0, float("-inf"))
            if getattr(mod, "attention_sinks", False):
                sink = mod.sinks.view(1, -1, 1, 1).expand(B, -1, T, 1)
                att = F.softmax(torch.cat([att, sink.to(att.dtype)], dim=-1), dim=-1)
                sink_sum[l] += att[..., 1:, -1].sum(dim=(0, 2)).double().cpu()
                att = att[..., :-1]
            else:
                att = F.softmax(att, dim=-1)
            prev_sum[l] += att[..., 1:, :].diagonal(dim1=-2, dim2=-1).sum(dim=(0, 2)).double().cpu()
            self_sum[l] += att.diagonal(dim1=-2, dim2=-1)[..., 1:].sum(dim=(0, 2)).double().cpu()
            if l == 0:
                n_q += B * (T - 1)
        return hook

    hooks = [model.h[l].attn.register_forward_pre_hook(make_hook(l)) for l in range(L)]
    with torch.no_grad():
        for b in batches:
            model(b.to(dev))
    for h in hooks:
        h.remove()
    model.to("cpu")
    return (prev_sum / n_q).numpy(), (self_sum / n_q).numpy(), (sink_sum / n_q).numpy(), n_q


def show(name, prev, slf, snk, n_q, has_sink):
    print(f"=== {name}  ({n_q} queries) ===")
    tables = [("prev-token mass att[i,i-1]", prev), ("self mass att[i,i]", slf)]
    if has_sink:
        tables.append(("sink-slot mass", snk))
    for title, m in tables:
        print(title)
        print("layer  " + "  ".join(f"h{h}    " for h in range(m.shape[1])))
        for l in range(m.shape[0]):
            print(f"  {l}    " + "  ".join(f"{m[l,h]:.3f}" for h in range(m.shape[1])))
        print()


pile_batches = [torch.stack(pile_rows[i:i+16]) for i in range(0, 500, 16)]
story_batches = [s[:512].unsqueeze(0) for s in story_rows]  # variable lengths -> one at a time

out = {}
for name, loader, batches, has_sink in [
    ("pile_4l", lambda: load.load_pile_4l()[0], pile_batches, False),
    ("sink45", lambda: load.load_sink(45)[0], pile_batches, True),
    ("sink46", lambda: load.load_sink(46)[0], pile_batches, True),
    ("simple_2l", lambda: load.load_simple_2l()[0], story_batches, False),
]:
    prev, slf, snk, n_q = run(loader(), batches)
    show(name, prev, slf, snk, n_q, has_sink)
    out[f"{name}|prev"], out[f"{name}|self"], out[f"{name}|sink"] = prev, slf, snk
    out[f"{name}|n_q"] = np.array(n_q)

(HERE / "cache").mkdir(exist_ok=True)
np.savez(HERE / "cache" / "attn_stats.npz", **out)
print("saved", HERE / "cache" / "attn_stats.npz")
