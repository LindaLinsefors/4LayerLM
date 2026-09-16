"""The actual sink test: where does attention probability mass go?

Recomputes per-head attention matrices A (the model runs flash attention, so
they are never materialized) from each attn module's captured input, using the
module's own projections/rotary buffers, and accumulates:

  1. mass on key 0 per (layer, head), and as a function of query position
  2. mass on mid-sequence <|endoftext|> keys (from queries after them),
     vs a position-matched baseline: the same key positions in EOS-free rows,
     and the uniform value E[1/(q+1)]
  3. hand-off: mass on key 0 for queries with vs without an EOS earlier in
     their window
  4. the value side of the sink signature: ||W_O^h v_p|| — the norm of what
     key p actually writes into the stream through head h (computed as
     sqrt(v^T W_O^hT W_O^h v)) — at position 0 and at EOS keys vs the
     all-position mean. A true sink has high mass AND small value output.

Position 0 here is the first token of the 512-token *chunk* (arbitrary
mid-document token in ~75% of rows), not a document start. Data: first N_ROWS
cached Pile rows. EOS at position 0 is excluded from the EOS-key stats (that's
the position-0 story).

Outputs: endoftext-pos0/attention_mass.png + printed tables.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 500
BATCH_SIZE = 8
EOS_ID = 0
Q_MIN = 64          # summary statistics use queries at positions >= Q_MIN
T = 512


def attn_probs(attn, x):
    """(B, H, T, T) attention probabilities, replicating the module's forward."""
    B, T_, C = x.shape
    q = attn.q_proj(x).view(B, T_, attn.n_head, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(B, T_, attn.n_key_value_heads, attn.head_dim).transpose(1, 2)
    pos = torch.arange(T_, device=x.device).unsqueeze(0)
    cos = attn.rotary_cos[pos].to(q.dtype)
    sin = attn.rotary_sin[pos].to(q.dtype)
    q, k = attn._apply_rotary_pos_emb(q, k, cos, sin)
    if attn.repeat_kv_heads > 1:
        k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
    att = (q @ k.transpose(-2, -1)) / np.sqrt(attn.head_dim)
    att = att.masked_fill(attn.bias[:, :, :T_, :T_] == 0, float("-inf"))
    return F.softmax(att, dim=-1)


rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]

model, _, _ = load_pile_4l()
model = model.to(DEVICE)
L, H = len(model.h), model.h[0].attn.n_head

caps: dict[int, torch.Tensor] = {}
hooks = [model.h[l].attn.register_forward_pre_hook(
    lambda _m, inp, l=l: caps.__setitem__(l, inp[0])) for l in range(L)]

on0_sum = np.zeros((2, L, H, T))     # [has-EOS-before?, l, h, q] mass on key 0
on0_cnt = np.zeros((2, T))           # rows contributing, per (cond, q)
eos_sum = np.zeros((L, H))           # mass on mid-seq EOS keys (queries after)
eos_n = 0
eos_ps = []                          # EOS key positions (for matched baselines)
base_sum = np.zeros((L, H, T))       # colmean over EOS-free rows, per key pos
base_cnt = 0                         # EOS-free rows
vout_all = np.zeros((L, H))          # sum of ||W_O^h v_p|| over all positions
vout_pos0 = np.zeros((L, H))         # ... at position 0
vout_eos = np.zeros((L, H))          # ... at mid-seq EOS keys
vout_n = 0
vout_eos_n = 0

# per-head Gram matrices of the o_proj column blocks: G_h = W_O^hT W_O^h
grams = []
for l in range(L):
    attn = model.h[l].attn
    WO = attn.o_proj.weight.detach()          # (768, 768), columns = value dims
    hd = attn.head_dim
    grams.append(torch.stack([
        WO[:, h * hd:(h + 1) * hd].T @ WO[:, h * hd:(h + 1) * hd] for h in range(H)
    ]))                                        # (H, hd, hd)

with torch.no_grad():
    for start in range(0, len(rows), BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        tok = batch.cpu().numpy()
        B = len(tok)
        mid_eos = (tok == EOS_ID).copy()
        mid_eos[:, 0] = False                       # position-0 EOS excluded
        before = np.zeros((B, T), dtype=bool)       # EOS strictly before query q
        before[:, 1:] = np.cumsum(mid_eos, axis=1)[:, :-1] > 0
        eos_free = ~mid_eos.any(axis=1)

        for l in range(L):
            A = attn_probs(model.h[l].attn, caps[l]).float()
            m0 = A[:, :, :, 0].cpu().numpy()        # (B, H, T) mass on key 0
            colsum = A.sum(dim=2).cpu().numpy()     # (B, H, T) per-key column sums
            diag = A.diagonal(dim1=2, dim2=3).cpu().numpy()
            with np.errstate(invalid="ignore"):
                colmean = (colsum - diag) / (T - 1 - np.arange(T))  # mean over q > p

            for cond in (0, 1):
                sel = before == bool(cond)          # (B, T)
                on0_sum[cond, l] += np.einsum("bhq,bq->hq", m0, sel)
                if l == 0:
                    on0_cnt[cond] += sel.sum(axis=0)
            b_idx, p_idx = np.nonzero(mid_eos[:, :T - 1])
            eos_sum[l] += colmean[b_idx, :, p_idx].sum(axis=0)
            base_sum[l] += colmean[eos_free].sum(axis=0)
            if l == 0:
                eos_ps.extend(p_idx.tolist())
                base_cnt += eos_free.sum()

            attn = model.h[l].attn
            x = caps[l]
            v = attn.v_proj(x).view(B, T, attn.n_key_value_heads,
                                    attn.head_dim).transpose(1, 2)
            if attn.repeat_kv_heads > 1:
                v = v.repeat_interleave(attn.repeat_kv_heads, dim=1)
            outn = torch.sqrt(torch.einsum(
                "bhtc,hcd,bhtd->bht", v, grams[l].to(v.dtype), v)).cpu().numpy()
            vout_all[l] += outn.sum(axis=(0, 2))
            vout_pos0[l] += outn[:, :, 0].sum(axis=0)
            be, pe = np.nonzero(mid_eos)
            vout_eos[l] += outn[be, :, pe].sum(axis=0)
        eos_n += mid_eos[:, :T - 1].sum()
        vout_n += B * T
        vout_eos_n += mid_eos.sum()
for h in hooks:
    h.remove()

eos_ps = np.array(eos_ps)
on0 = on0_sum / on0_cnt[:, None, None, :]                      # (2, L, H, T)
mass0 = (on0_sum[:, :, :, Q_MIN:].sum(3).sum(0)
         / on0_cnt[:, Q_MIN:].sum())                           # (L, H), all queries
mass0_cond = on0_sum[:, :, :, Q_MIN:].sum(3) / on0_cnt[:, Q_MIN:].sum(1)[:, None, None]
eos_mass = eos_sum / eos_n                                     # (L, H)
base_col = base_sum / base_cnt                                 # (L, H, T)
base_matched = base_col[:, :, eos_ps].mean(axis=2)             # (L, H)
uni = float(np.mean([1 / np.arange(p + 2, T + 1) for p in eos_ps[:0]] or [0]))
uni = np.mean([np.mean(1 / np.arange(p + 2, T + 1)) for p in eos_ps])

print(f"{N_ROWS} rows; {eos_n} mid-seq EOS keys; {base_cnt} EOS-free rows; "
      f"uniform baseline E[1/(q+1)] = {uni:.4f}")
print(f"\nmass on key 0 (queries >= {Q_MIN}), layer x head:")
for l in range(L):
    print(f"  L{l}: " + "  ".join(f"h{h}:{mass0[l, h]:.3f}" for h in range(H))
          + f"   | layer mean {mass0[l].mean():.3f}")
print(f"  uniform baseline per key ~ {np.mean(1 / np.arange(Q_MIN + 1, T + 1)):.4f}")
print("\nmass on mid-seq EOS keys vs position-matched baseline (EOS-free rows):")
for l in range(L):
    print(f"  L{l}: EOS " + "  ".join(f"h{h}:{eos_mass[l, h]:.4f}" for h in range(H)))
    print(f"      base " + "  ".join(f"h{h}:{base_matched[l, h]:.4f}" for h in range(H)))
print("\nhand-off: mass on key 0 (queries >= 64), no EOS before vs EOS before:")
for l in range(L):
    print(f"  L{l}: no-EOS {mass0_cond[0, l].mean():.3f}  after-EOS "
          f"{mass0_cond[1, l].mean():.3f}   per head after/no ratio: "
          + "  ".join(f"h{h}:{mass0_cond[1, l, h] / mass0_cond[0, l, h]:.2f}"
                      for h in range(H)))

vo_all = vout_all / vout_n
vo_p0 = vout_pos0 / N_ROWS
vo_eos = vout_eos / vout_eos_n
print("\nvalue-output norm ||W_O^h v_p||, ratio to all-position mean:")
for l in range(L):
    print(f"  L{l}: pos0/all " + "  ".join(
        f"h{h}:{vo_p0[l, h] / vo_all[l, h]:.2f}" for h in range(H)))
    print(f"      EOS/all  " + "  ".join(
        f"h{h}:{vo_eos[l, h] / vo_all[l, h]:.2f}" for h in range(H)))

# ---- figure ----
fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
(ax_hm, ax_q), (ax_eos, ax_ho), (ax_v, ax_eff) = axes

im = ax_hm.imshow(mass0, cmap="viridis", aspect="auto")
for l in range(L):
    for h in range(H):
        ax_hm.text(h, l, f"{mass0[l, h]:.2f}", ha="center", va="center",
                   fontsize=8, color="w" if mass0[l, h] < mass0.max() * 0.6 else "k")
ax_hm.set_xticks(range(H), [f"h{h}" for h in range(H)])
ax_hm.set_yticks(range(L), [f"L{l}" for l in range(L)])
ax_hm.set_title(f"mean attention mass on key 0 (queries ≥ {Q_MIN})", fontsize=11)
fig.colorbar(im, ax=ax_hm, shrink=0.8)

qs = np.arange(1, T)
on0_all = on0_sum.sum(0) / on0_cnt.sum(0)      # (L, H, T) mean over all rows
for l in range(L):
    ax_q.plot(qs, on0_all[l, :, 1:].mean(0), lw=1.2, label=f"layer {l + 1}")
ax_q.plot(qs, 1 / (qs + 1), "k--", lw=1, label="uniform 1/(q+1)")
ax_q.set_yscale("log")
ax_q.set_xlabel("query position q")
ax_q.set_ylabel("mean mass on key 0")
ax_q.set_title("mass on key 0 vs query position (mean over heads)", fontsize=11)
ax_q.legend(fontsize=8, frameon=False)

x = np.arange(L * H)
ax_eos.bar(x - 0.2, eos_mass.flatten(), 0.4, label="mid-seq EOS key", color="tab:red")
ax_eos.bar(x + 0.2, base_matched.flatten(), 0.4,
           label="same positions, EOS-free rows", color="tab:blue")
ax_eos.axhline(uni, color="k", ls="--", lw=1, label="uniform")
ax_eos.set_xticks(x, [f"L{l}h{h}" for l in range(L) for h in range(H)],
                  rotation=90, fontsize=7)
ax_eos.set_yscale("log")
ax_eos.set_ylabel("mean mass on key (queries after it)")
ax_eos.set_title("attention mass on mid-sequence <|endoftext|> keys", fontsize=11)
ax_eos.legend(fontsize=8, frameon=False)

ax_ho.bar(x - 0.2, mass0_cond[0].flatten(), 0.4, label="queries with no EOS before",
          color="tab:blue")
ax_ho.bar(x + 0.2, mass0_cond[1].flatten(), 0.4, label="queries after an EOS",
          color="tab:red")
ax_ho.set_xticks(x, [f"L{l}h{h}" for l in range(L) for h in range(H)],
                 rotation=90, fontsize=7)
ax_ho.set_ylabel(f"mass on key 0 (queries ≥ {Q_MIN})")
ax_ho.set_title("hand-off test: does an EOS in context reduce the position-0 sink?",
                fontsize=11)
ax_ho.legend(fontsize=8, frameon=False)

ax_v.bar(x - 0.27, vo_p0.flatten(), 0.27, label="position 0", color="tab:orange")
ax_v.bar(x, vo_eos.flatten(), 0.27, label="mid-seq EOS", color="tab:red")
ax_v.bar(x + 0.27, vo_all.flatten(), 0.27, label="all positions (mean)",
         color="tab:blue")
ax_v.set_xticks(x, [f"L{l}h{h}" for l in range(L) for h in range(H)],
                rotation=90, fontsize=7)
ax_v.set_yscale("log")
ax_v.set_ylabel("mean ‖W_O$^h$ v$_p$‖")
ax_v.set_title("value-path output norm: what the key writes if attended to",
               fontsize=11)
ax_v.legend(fontsize=8, frameon=False)

eff0 = mass0 * vo_p0
eff_typ = vo_all * np.mean(1 / np.arange(Q_MIN + 1, T + 1))
ax_eff.bar(x - 0.2, eff0.flatten(), 0.4,
           label="key 0: mass × ‖W_O v₀‖", color="tab:orange")
ax_eff.bar(x + 0.2, eff_typ.flatten(), 0.4,
           label="typical key: uniform mass × mean ‖W_O v‖", color="tab:blue")
ax_eff.set_xticks(x, [f"L{l}h{h}" for l in range(L) for h in range(H)],
                  rotation=90, fontsize=7)
ax_eff.set_yscale("log")
ax_eff.set_ylabel("effective write norm")
ax_eff.set_title("does the position-0 mass actually write anything? "
                 "(mass × value norm)", fontsize=11)
ax_eff.legend(fontsize=8, frameon=False)

for ax in (ax_q, ax_eos, ax_ho, ax_v, ax_eff):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — attention-sink test: probability mass on position 0 vs "
             f"<|endoftext|> ({N_ROWS} Pile rows)", y=0.995)
fig.tight_layout()
fig.savefig(HERE.parent / "attention_mass.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {HERE.parent / 'attention_mass.png'}")
