"""How does attention 1 create the position-0 (early-position) signal?

Attention is the only place position information can enter this model (no
positional embeddings; RoPE acts only inside q·k). At query position p a head's
output is a convex mixture over keys 0..p; at p = 0 the softmax has one key,
so the output is exactly the token's own value, out_h = W_O^h v_self. A head
whose bulk attention profile is position- or content-selective therefore
produces a *different* output distribution at small p — that difference is the
linear signal the probes in pos0_probe.py read.

Per layer-0 head this script measures:
  1. bulk attention profile by relative offset (self, -1, -2, -3, -4..-7,
     -8..-31, -32..-127, <=-128), queries >= 128
  2. mean self-attention mass as a function of query position p (0..64) —
     how fast the forced self-attention relaxes
  3. per-head linear readability: LDA probe AUC for "position p vs bulk" using
     only head h's value-mixture y_h (128-d, equivalent to probing its
     residual write W_O^h y_h), for p in {0, 1, 4, 16}
  4. per-head mean-difference norm ||E[W_O^h y_h | p] - E[W_O^h y_h | bulk]||
     and its share of the full attention-output mean difference

Data: first N_ROWS cached Pile rows, positions EARLY + 6 random bulk >= 128
per row (same layout as pos0_probe.py, fresh forward pass — layer 0 only is
needed but the full model runs anyway, ~1 min).
Output: endoftext-pos0/pos0_attn1_mechanism.png + printed tables.
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
N_ROWS = 1000
BATCH_SIZE = 16
EOS_ID = 0
T = 512
EARLY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64]
N_BULK = 6
BULK_MIN = 128
N_FOLDS = 5
LAM = 3e-2
PROBE_PS = [0, 1, 4, 16]
OFFSET_BINS = [(0, 0, "self"), (1, 1, "-1"), (2, 2, "-2"), (3, 3, "-3"),
               (4, 7, "-4..-7"), (8, 31, "-8..-31"), (32, 127, "-32..-127"),
               (128, T, "<=-128")]


def attn_probs(attn, x):
    B, T_, _ = x.shape
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
rng = np.random.default_rng(0)
bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))

model, _, _ = load_pile_4l()
model = model.to(DEVICE)
attn0 = model.h[0].attn
H, hd = attn0.n_head, attn0.head_dim

caps: dict[str, torch.Tensor] = {}
hooks = [model.h[0].attn.register_forward_pre_hook(
    lambda _m, inp: caps.__setitem__("x", inp[0]))]

n_keep = len(EARLY) + N_BULK
y_keep = np.zeros((N_ROWS, n_keep, 768), dtype=np.float16)   # pre-o_proj value mix
kept_tok = np.zeros((N_ROWS, n_keep), dtype=np.int64)
off_sum = np.zeros((H, len(OFFSET_BINS)))                    # bulk offset profile
off_n = 0
self_vs_p = np.zeros((H, 65))                                # mean self mass at p=0..64
self_n = 0

with torch.no_grad():
    for start in range(0, len(rows), BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        x = caps["x"]                                        # rms_1 output
        B = len(batch)
        A = attn_probs(attn0, x)                             # (B, H, T, T)

        v = attn0.v_proj(x).view(B, T, attn0.n_key_value_heads, hd).transpose(1, 2)
        if attn0.repeat_kv_heads > 1:
            v = v.repeat_interleave(attn0.repeat_kv_heads, dim=1)
        y = (A @ v).transpose(1, 2).reshape(B, T, 768)       # heads concatenated

        tok = batch.cpu().numpy()
        for bi in range(B):
            ri = start + bi
            pos = np.concatenate([np.array(EARLY), bulk_pos[ri]])
            kept_tok[ri] = tok[bi, pos]
            y_keep[ri] = y[bi, pos].float().cpu().numpy()

        qs = torch.arange(BULK_MIN, T, device=DEVICE)
        Abulk = A[:, :, qs, :]                               # (B, H, nq, T)
        offs = qs[:, None] - torch.arange(T, device=DEVICE)[None, :]  # (nq, T)
        for bi_, (lo, hi, _) in enumerate(OFFSET_BINS):
            m = ((offs >= lo) & (offs <= hi)).to(A.dtype)
            off_sum[:, bi_] += (Abulk * m[None, None]).sum((0, 2, 3)).cpu().numpy()
        off_n += B * len(qs)
        diag = A.diagonal(dim1=2, dim2=3)                    # (B, H, T)
        self_vs_p += diag[:, :, :65].sum(0).cpu().numpy()
        self_n += B
for h in hooks:
    h.remove()

off_prof = off_sum / off_n
self_vs_p /= self_n
WO = model.h[0].attn.o_proj.weight.detach().cpu().numpy()    # (768, 768)
del model

print("bulk attention profile by relative offset (queries >= 128), per head:")
labels = [b[2] for b in OFFSET_BINS]
print("head".ljust(6) + "".join(lab.rjust(10) for lab in labels))
for h in range(H):
    print(f"h{h}".ljust(6) + "".join(f"{off_prof[h, b]:10.3f}" for b in range(len(OFFSET_BINS))))

# ------------------------------------------------------------------- probing
def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


folds = np.arange(N_ROWS) % N_FOLDS
bulk_slots = slice(len(EARLY), None)
bulk_ok = kept_tok[:, bulk_slots] != EOS_ID

def probe_auc(feats: np.ndarray, pi: int) -> float:
    """CV AUC for position EARLY[pi] vs bulk using feature array (rows, keep, d)."""
    d = feats.shape[-1]
    ok1 = kept_tok[:, pi] != EOS_ID
    s0s, s1s = [], []
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        Xb = feats[tr][:, bulk_slots][bulk_ok[tr]]
        mu_b = Xb.mean(0)
        Xc = Xb - mu_b
        Sig = Xc.T @ Xc / (len(Xb) - 1)
        mu_p = feats[tr & ok1, pi].mean(0)
        w = np.linalg.solve(Sig + LAM * np.trace(Sig) / d * np.eye(d), mu_p - mu_b)
        s0s.append(feats[te][:, bulk_slots][bulk_ok[te]] @ w)
        s1s.append(feats[te & ok1, pi] @ w)
    return auc(np.concatenate(s0s), np.concatenate(s1s))


Y = y_keep.astype(np.float64)
head_auc = np.zeros((H, len(PROBE_PS)))
for h in range(H):
    for pj, p in enumerate(PROBE_PS):
        head_auc[h, pj] = probe_auc(Y[:, :, h * hd:(h + 1) * hd], EARLY.index(p))
full_auc = [probe_auc(Y, EARLY.index(p)) for p in PROBE_PS]

print("\nper-head LDA probe AUC on the head's value mixture y_h (128-d):")
print("head".ljust(6) + "".join(f"p={p}".rjust(9) for p in PROBE_PS))
for h in range(H):
    print(f"h{h}".ljust(6) + "".join(f"{head_auc[h, pj]:9.3f}" for pj in range(len(PROBE_PS))))
print("all".ljust(6) + "".join(f"{full_auc[pj]:9.3f}" for pj in range(len(PROBE_PS))))

# per-head mean-difference contribution in residual space
mu_bulk = Y[:, bulk_slots][bulk_ok].mean(0)
delta_full = None
print("\nper-head residual-space mean difference ||W_O^h (E[y_h|p] - E[y_h|bulk])||:")
print("head".ljust(6) + "".join(f"p={p}".rjust(9) for p in PROBE_PS))
head_dn = np.zeros((H, len(PROBE_PS)))
for pj, p in enumerate(PROBE_PS):
    pi = EARLY.index(p)
    ok1 = kept_tok[:, pi] != EOS_ID
    dmu = Y[ok1, pi].mean(0) - mu_bulk                      # (768,) in y space
    for h in range(H):
        head_dn[h, pj] = np.linalg.norm(WO[:, h * hd:(h + 1) * hd] @ dmu[h * hd:(h + 1) * hd])
for h in range(H):
    print(f"h{h}".ljust(6) + "".join(f"{head_dn[h, pj]:9.4f}" for pj in range(len(PROBE_PS))))
print("full".ljust(6) + "".join(
    f"{np.linalg.norm(WO @ (Y[kept_tok[:, EARLY.index(p)] != EOS_ID, EARLY.index(p)].mean(0) - mu_bulk)):9.4f}"
    for p in PROBE_PS))

# ---------------------------------------------------------------------- plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax_off, ax_self), (ax_auc, ax_dn) = axes
colors = plt.cm.tab10(np.arange(H))

xb = np.arange(len(OFFSET_BINS))
wbar = 0.13
for h in range(H):
    ax_off.bar(xb + (h - (H - 1) / 2) * wbar, off_prof[h], wbar,
               color=colors[h], label=f"h{h}")
ax_off.set_xticks(xb, labels)
ax_off.set_xlabel("relative offset (key − query)")
ax_off.set_ylabel("mean attention mass")
ax_off.set_title("layer-0 bulk attention profile by offset (queries ≥ 128)", fontsize=11)
ax_off.legend(fontsize=8, frameon=False, ncol=2)

for h in range(H):
    ax_self.plot(np.arange(65), self_vs_p[h], color=colors[h], lw=1.3, label=f"h{h}")
ax_self.axhline(1.0, color="k", ls=":", lw=0.8)
ax_self.set_xlabel("query position p")
ax_self.set_ylabel("mean self-attention mass")
ax_self.set_title("forced self-attention: mass on own key vs position\n"
                  "(= 1 at p = 0 by softmax over one key)", fontsize=11)
ax_self.legend(fontsize=8, frameon=False, ncol=2)

xh = np.arange(H + 1)
wbar = 0.19
for pj, p in enumerate(PROBE_PS):
    vals = np.append(head_auc[:, pj], full_auc[pj])
    ax_auc.bar(xh + (pj - (len(PROBE_PS) - 1) / 2) * wbar, vals, wbar,
               label=f"p={p}")
ax_auc.axhline(0.5, color="k", ls=":", lw=0.8)
ax_auc.set_xticks(xh, [f"h{h}" for h in range(H)] + ["all"])
ax_auc.set_ylim(0.45, 1.0)
ax_auc.set_ylabel("CV probe AUC")
ax_auc.set_title("which heads carry the signal: probe on each head's\n"
                 "value mixture y_h alone (LDA, 5-fold CV)", fontsize=11)
ax_auc.legend(fontsize=8, frameon=False)

for pj, p in enumerate(PROBE_PS):
    ax_dn.bar(np.arange(H) + (pj - (len(PROBE_PS) - 1) / 2) * wbar, head_dn[:, pj],
              wbar, label=f"p={p}")
ax_dn.set_xticks(np.arange(H), [f"h{h}" for h in range(H)])
ax_dn.set_ylabel("‖W_O$^h$ (E[y$_h$|p] − E[y$_h$|bulk])‖")
ax_dn.set_title("per-head mean-difference write into the residual stream", fontsize=11)
ax_dn.legend(fontsize=8, frameon=False)

for ax in axes.flat:
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — how layer-0 attention creates the early-position signal "
             f"({N_ROWS} Pile rows)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_attn1_mechanism.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {HERE.parent / 'pos0_attn1_mechanism.png'}")
