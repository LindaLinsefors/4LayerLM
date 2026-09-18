"""Do the sink-heavy L0 heads compensate with a large OV circuit?  (sink seed 45)

Per head h, over cached Pile rows:
  static:  singular values of the OV map  M_h = W_O^h W_V^h  (768x768, rank <= 128)
  per key: ||W_O^h v_j||  -- the residual write a key would cause at attention 1
  per query (positions >= 64): content mass  c_i = 1 - A_{i,sink}  and actual write norm
           ||w_i^h|| = ||W_O^h sum_j A_ij v_j||  (the sink slot contributes zero value)
Caches log-histograms of ||w||, ||W_O v||, plus mean ||w|| binned by content mass, means.
Writes hide/cache/ov_stats.npz.  ~1 min local GPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import load  # noqa: E402

N_ROWS = 500
BATCH = 8
T = 512
QPOS_MIN = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NB = 120                       # log bins for norms
LOG_LO, LOG_HI = -4.0, 2.0     # ||.|| bin range 1e-4 .. 1e2
NC = 20                        # content-mass bins


def main():
    model, _ = load.load_sink(45)
    model.to(DEV)
    attn = model.h[0].attn
    rms1 = model.h[0].rms_1
    nh, hd = model.config.n_head, attn.head_dim

    W_O = attn.o_proj.weight           # (768, 768)
    W_V = attn.v_proj.weight           # (768, 768)
    WO_h = W_O.view(768, nh, hd).permute(1, 0, 2)      # (nh, 768, 128)
    sv = [torch.linalg.svdvals(WO_h[h] @ W_V[h * hd:(h + 1) * hd]).cpu().numpy()
          for h in range(nh)]
    sv = np.stack(sv)                  # (nh, 768); only first 128 nonzero

    rows = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")
    rows = torch.stack([r[:T] for r in rows[:N_ROWS]])

    wnorm_hist = torch.zeros(nh, NB, dtype=torch.float64, device=DEV)
    vnorm_hist = torch.zeros(nh, NB, dtype=torch.float64, device=DEV)
    cmass_wsum = torch.zeros(nh, NC, dtype=torch.float64, device=DEV)  # sum ||w|| per c bin
    cmass_cnt = torch.zeros(nh, NC, dtype=torch.float64, device=DEV)
    wnorm_sum = torch.zeros(nh, dtype=torch.float64, device=DEV)
    vnorm_sum = torch.zeros(nh, dtype=torch.float64, device=DEV)
    xnorm_sum = torch.zeros((), dtype=torch.float64, device=DEV)  # residual (wte) norms
    n_q = 0
    n_k = 0

    causal = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos = attn.rotary_cos[:T].to(DEV)
    sin = attn.rotary_sin[:T].to(DEV)

    for start in range(0, N_ROWS, BATCH):
        toks = rows[start : start + BATCH].to(DEV)
        B = toks.shape[0]
        with torch.no_grad():
            emb = model.wte(toks)
            xnorm_sum += emb.norm(dim=-1).double().sum()
            x = rms1(emb)
            q = attn.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
            v = attn.v_proj(x).view(B, T, nh, hd).transpose(1, 2)
            q = q * cos + attn._rotate_every_two(q) * sin
            k = k * cos + attn._rotate_every_two(k) * sin
            logits = (q @ k.transpose(-2, -1)) / hd**0.5
            logits = logits.masked_fill(~causal, float("-inf"))
            snk = attn.sinks.view(1, nh, 1, 1).expand(B, nh, T, 1)
            att = torch.softmax(torch.cat([logits, snk], dim=-1), dim=-1)

            # per-key residual write ||W_O^h v_j||
            vw = torch.einsum("hdc,bhtc->bhtd", WO_h, v)      # (B,nh,T,768)
            vn = vw.norm(dim=-1)                              # (B,nh,T)
            vnorm_sum += vn.double().sum((0, 2)) / 1          # per head
            n_k += B * T

            # per-query content write
            y = att[..., :T] @ v                              # (B,nh,T,128)
            w = torch.einsum("hdc,bhtc->bhtd", WO_h, y)
            wn = w.norm(dim=-1)[:, :, QPOS_MIN:]              # (B,nh,T-64)
            c = 1.0 - att[:, :, QPOS_MIN:, -1]                # content mass
            wnorm_sum += wn.double().sum((0, 2))
            n_q += B * (T - QPOS_MIN)

            logw = wn.clamp_min(10**LOG_LO).log10()
            logv = vn.clamp_min(10**LOG_LO).log10()
            for h in range(nh):
                wnorm_hist[h] += torch.histc(logw[:, h], bins=NB, min=LOG_LO, max=LOG_HI)
                vnorm_hist[h] += torch.histc(logv[:, h], bins=NB, min=LOG_LO, max=LOG_HI)
                cb = (c[:, h] * NC).long().clamp(max=NC - 1).reshape(-1)
                cmass_wsum[h].index_add_(0, cb, wn[:, h].reshape(-1).double())
                cmass_cnt[h].index_add_(0, cb, torch.ones_like(cb, dtype=torch.float64))
        print(f"{start + B}/{N_ROWS}", end="\r")

    out = HERE / "cache"
    np.savez_compressed(
        out / "ov_stats.npz",
        sv=sv[:, :128],
        wnorm_hist=wnorm_hist.cpu().numpy(), vnorm_hist=vnorm_hist.cpu().numpy(),
        log_lo=LOG_LO, log_hi=LOG_HI,
        cmass_wsum=cmass_wsum.cpu().numpy(), cmass_cnt=cmass_cnt.cpu().numpy(),
        wnorm_mean=(wnorm_sum / n_q).cpu().numpy(),
        vnorm_mean=(vnorm_sum / n_k).cpu().numpy(),
        emb_norm_mean=(xnorm_sum / n_k).cpu().numpy(),
        n_rows=N_ROWS,
    )
    print(f"\nsaved {out / 'ov_stats.npz'}")
    print("top OV singular value per head:", sv[:, 0].round(3))
    print("OV Frobenius per head:", np.sqrt((sv**2).sum(1)).round(3))
    print("mean ||W_O v|| per head:", (vnorm_sum / n_k).cpu().numpy().round(3))
    print("mean ||w|| per head:", (wnorm_sum / n_q).cpu().numpy().round(4))
    print("mean ||emb||:", float(xnorm_sum / n_k))


if __name__ == "__main__":
    main()
