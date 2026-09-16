"""Per-head version of qk28_scores_act.py: one heatmap per attention head.

score_h(c, D) = s_q s_c (R_D uhat_q,h) . uhat_c,h / sqrt(d_head)

with uhat = U/||U|| normalized over the full 768-dim output vector, so the six
per-head panels sum exactly to the total map in qk28_scores_act.png. Activation
strengths s (signed mean on CI > 0.1 tokens) come from cache/act_stats.npz
(computed by qk28_scores_act.py).

Writes ../qk28_scores_act_heads.png; caches per-head unit scores in
cache/qk28_scores_normU_heads.npz.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

N_HEADS, D_HEAD, N_CTX, ROPE_BASE = 6, 128, 512, 10000
CACHE = HERE / "cache" / "qk28_scores_normU_heads.npz"


def compute() -> tuple[np.ndarray, np.ndarray]:
    """Return (alive k ids, per-head unit scores (n_alive, 6, N_CTX))."""
    from load import PILE_4L, ParameterComponents

    pc = ParameterComponents(PILE_4L / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")
    U_q = pc.components["h.0.attn.q_proj"].U.float().numpy()
    U_k = pc.components["h.0.attn.k_proj"].U.float().numpy()

    mean_ci = np.load(ROOT / "pile-qk-comps" / "hide" / "cache" / "mean_ci.npz")
    k_ids = np.where(mean_ci["h.0.attn.k_proj"] > 1e-6)[0]

    q = (U_q[28] / np.linalg.norm(U_q[28])).reshape(N_HEADS, D_HEAD)
    K = (U_k[k_ids] / np.linalg.norm(U_k[k_ids], axis=1, keepdims=True)).reshape(-1, N_HEADS, D_HEAD)

    half = D_HEAD // 2
    freq = ROPE_BASE ** (np.arange(half) / half)
    theta = np.arange(N_CTX)[:, None] / freq[None, :]
    cos, sin = np.tile(np.cos(theta), 2), np.tile(np.sin(theta), 2)

    q_rh = np.concatenate([-q[:, half:], q[:, :half]], axis=1)
    q_rot = q[None] * cos[:, None] + q_rh[None] * sin[:, None]      # (512, 6, 128)

    scores = np.einsum("dhi,chi->chd", q_rot, K) / np.sqrt(D_HEAD)  # (n, 6, 512)
    return k_ids, scores.astype(np.float32)


if CACHE.exists():
    d = np.load(CACHE)
    k_ids, unit_h = d["k_ids"], d["scores"]
else:
    k_ids, unit_h = compute()
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, k_ids=k_ids, scores=unit_h)

st = np.load(HERE / "cache" / "act_stats.npz")
scores = float(st["s_q"]) * st["s_k_signed"][:, None, None] * unit_h  # (n, 6, 512)

# consistency with the summed map
total = np.load(HERE / "cache" / "qk28_scores_normU.npz")["scores"]
assert np.allclose(unit_h.sum(axis=1), total, atol=1e-5)

n = len(k_ids)
vmax = np.abs(scores).max()
print(f"max |per-head score| = {vmax:.2f} logits "
      f"(head of max: h{np.unravel_index(np.abs(scores).argmax(), scores.shape)[1]})")

fig, axes = plt.subplots(1, N_HEADS, figsize=(26, 15), sharey=True)
for h, ax in enumerate(axes):
    im = ax.imshow(scores[:, h], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=(-0.5, N_CTX - 0.5, n - 0.5, -0.5), interpolation="nearest")
    ax.set_title(f"head {h}", fontsize=12)
    ax.set_xlabel("distance D")
    ax.tick_params(labelsize=8)
axes[0].set_yticks(np.arange(n))
axes[0].set_yticklabels(k_ids, fontsize=4.5)
axes[0].set_ylabel("alive k component (id order)")
fig.suptitle("Per-head typical pre-softmax score contribution  "
             "$s_{q28}\\, s_c (R_D \\hat u_{q,h})\\cdot\\hat u_{c,h}/\\sqrt{128}$"
             "   (panels sum to qk28_scores_act.png; shared color scale)", fontsize=13)
cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
cbar.set_label("score (logit units)")
out = HERE.parent / "qk28_scores_act_heads.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
