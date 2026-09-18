"""Layer-0 attention statistics for the sink seed 45 model (t-87f91319).

Recomputes the full per-head layer-0 attention (including the learned sink
slot, GPT-OSS style: softmax over [key logits, sink logit], sink column
dropped) over cached Pile rows.  Layer 0's attention input is exactly
rms_1(wte[tokens]) (pre-norm, first block), so no full forward pass is needed.
Uses load.load_sink(45) => corrected RoPE (see sink-models/rope_report.md).

Accumulates (per head h = 0..5):
  sink_pos[h, i]     mean sink mass at query position i
  key0_pos[h, i]     mean mass on key 0 at query position i
  neff_pos[h, i]     mean n_eff = exp(entropy) of the full (keys+sink) distribution
  offset[h, d]       mean mass att[i, i-d] over all queries (d = 0..511)
  offset_logit[h, d] mean pre-softmax logit q_i.k_{i-d}/sqrt(128) over all queries
  sink_qtok[h, t]    summed sink mass over queries with token id t   (positions >= 64)
  qtok_cnt[t]        number of such queries
  recv_ktok[h, t]    summed window-received mass over keys with token id t:
                     recv(j) = mean_{d=1..64} att[j+d, j], keys 1 <= j <= T-65
  ktok_cnt[t]        number of such keys
  sink_hist[h, b]    histogram (100 bins on [0,1]) of per-query sink mass,
                     queries at positions >= 64
  mass_dhist[h, d, b]  for d = 0..32: histogram of att[i, i-d] over queries,
                     320 log10 bins on [1e-8, 1] (values clamped up to 1e-8)
  logit_dhist[h, d, b] for d = 0..32: histogram of the logit, 450 bins on [-30, 15]
Writes attention-patherns/hide/cache/l0_attn_stats.npz.  ~1-2 min on local GPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import load  # noqa: E402

N_ROWS = 1000
BATCH = 16
T = 512
WINDOW = 64          # received-mass window (offsets 1..64 after the key)
QPOS_MIN = 64        # query-token sink stats use positions >= this
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model, tokenizer = load.load_sink(45)  # corrected RoPE
    model.to(DEV)
    attn = model.h[0].attn
    rms1 = model.h[0].rms_1
    V = model.config.vocab_size
    nh = model.config.n_head

    rows = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")
    rows = torch.stack([r[:T] for r in rows[:N_ROWS]])  # (N, 512)

    sink_pos = torch.zeros(nh, T, dtype=torch.float64, device=DEV)
    key0_pos = torch.zeros(nh, T, dtype=torch.float64, device=DEV)
    neff_pos = torch.zeros(nh, T, dtype=torch.float64, device=DEV)
    offset = torch.zeros(nh, T, dtype=torch.float64, device=DEV)
    offset_logit = torch.zeros(nh, T, dtype=torch.float64, device=DEV)
    sink_qtok = torch.zeros(nh, V, dtype=torch.float64, device=DEV)
    qtok_cnt = torch.zeros(V, dtype=torch.float64, device=DEV)
    recv_ktok = torch.zeros(nh, V, dtype=torch.float64, device=DEV)
    ktok_cnt = torch.zeros(V, dtype=torch.float64, device=DEV)
    N_BINS = 100
    sink_hist = torch.zeros(nh, N_BINS, dtype=torch.float64, device=DEV)
    D_MAX = 32
    NBM, NBL = 320, 450
    mass_dhist = torch.zeros(nh, D_MAX + 1, NBM, dtype=torch.float64, device=DEV)
    logit_dhist = torch.zeros(nh, D_MAX + 1, NBL, dtype=torch.float64, device=DEV)

    causal = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos = attn.rotary_cos[:T].to(DEV)
    sin = attn.rotary_sin[:T].to(DEV)

    for start in range(0, N_ROWS, BATCH):
        toks = rows[start : start + BATCH].to(DEV)
        B = toks.shape[0]
        with torch.no_grad():
            x = rms1(model.wte(toks))
            q = attn.q_proj(x).view(B, T, nh, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, nh, attn.head_dim).transpose(1, 2)
            q = q * cos + attn._rotate_every_two(q) * sin
            k = k * cos + attn._rotate_every_two(k) * sin
            logits = (q @ k.transpose(-2, -1)) / attn.head_dim**0.5
            logits = logits.masked_fill(~causal, float("-inf"))
            sink = attn.sinks.view(1, nh, 1, 1).expand(B, nh, T, 1)
            att = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)  # (B,nh,T,T+1)

            s = att[..., -1]                       # (B,nh,T) sink mass
            for h in range(nh):
                sink_hist[h] += torch.histc(s[:, h, QPOS_MIN:], bins=N_BINS, min=0.0, max=1.0)
            sink_pos += s.sum(0).double()
            key0_pos += att[..., 0].sum(0).double()
            p = att.clamp_min(1e-30)
            H = -(p * p.log()).sum(-1)
            neff_pos += H.exp().sum(0).double()

            content = att[..., :T]
            for d in range(T):
                diag = content.diagonal(offset=-d, dim1=2, dim2=3)  # (B,nh,T-d): att[i, i-d]
                offset[:, d] += diag.sum((0, 2)).double()
                ldiag = logits.diagonal(offset=-d, dim1=2, dim2=3)
                offset_logit[:, d] += ldiag.sum((0, 2)).double()
                if d <= D_MAX:
                    lmass = diag.clamp_min(1e-8).log10()
                    for h in range(nh):
                        mass_dhist[h, d] += torch.histc(lmass[:, h], bins=NBM, min=-8.0, max=0.0)
                        logit_dhist[h, d] += torch.histc(ldiag[:, h], bins=NBL, min=-30.0, max=15.0)
                if 1 <= d <= WINDOW:
                    # contribution to received mass of keys j = 0..T-1-d
                    recv_part = diag  # key j receives from query j+d
                    # accumulate only keys j in [1, T-1-WINDOW] (full window, skip key 0)
                    jmax = T - 1 - WINDOW
                    ids = toks[:, 1 : jmax + 1]
                    vals = recv_part[:, :, 1 : jmax + 1] / WINDOW
                    flat = ids.reshape(-1)
                    recv_ktok.index_add_(1, flat, vals.reshape(B, nh, -1).transpose(0, 1)
                                         .reshape(nh, -1).double())
            # key-token counts (once per batch, not per offset)
            ktok_cnt.index_add_(0, toks[:, 1 : T - WINDOW].reshape(-1),
                                torch.ones((T - 1 - WINDOW) * B, dtype=torch.float64, device=DEV))

            qids = toks[:, QPOS_MIN:].reshape(-1)
            sink_qtok.index_add_(1, qids, s[:, :, QPOS_MIN:].reshape(B, nh, -1)
                                 .transpose(0, 1).reshape(nh, -1).double())
            qtok_cnt.index_add_(0, qids, torch.ones_like(qids, dtype=torch.float64))
        print(f"{start + B}/{N_ROWS}", end="\r")

    n_q = N_ROWS  # rows per position
    counts_d = torch.tensor([N_ROWS * (T - d) for d in range(T)], dtype=torch.float64)

    out = HERE / "cache"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "l0_attn_stats.npz",
        sink_pos=(sink_pos / n_q).cpu().numpy(),
        key0_pos=(key0_pos / n_q).cpu().numpy(),
        neff_pos=(neff_pos / n_q).cpu().numpy(),
        offset=(offset.cpu() / counts_d).numpy(),
        offset_logit=(offset_logit.cpu() / counts_d).numpy(),
        sink_qtok=sink_qtok.cpu().numpy(),
        qtok_cnt=qtok_cnt.cpu().numpy(),
        recv_ktok=recv_ktok.cpu().numpy(),
        ktok_cnt=ktok_cnt.cpu().numpy(),
        sink_hist=sink_hist.cpu().numpy(),
        mass_dhist=mass_dhist.cpu().numpy(),
        logit_dhist=logit_dhist.cpu().numpy(),
        sink_logits=attn.sinks.detach().cpu().numpy(),
        n_rows=N_ROWS, window=WINDOW, qpos_min=QPOS_MIN,
    )
    print(f"\nsaved {out / 'l0_attn_stats.npz'}")
    print("L0 sink logits per head:", attn.sinks.detach().cpu().numpy().round(3))


if __name__ == "__main__":
    main()
