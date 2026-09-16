"""Pre-softmax attention scores between q:28 and every alive L0 k-component,
as a function of query-key distance -- first pass: from the components' output
vectors alone (no data).

Per component c and distance D (query pos i, key pos j, D = i - j):

    score(c, D) = sum_h (R_D q_h) . k_{c,h} / sqrt(d_head)

where q = U_q[28] / ||U_q[28]||, k_c = U_k[c] / ||U_k[c]||, reshaped to
(6 heads, 128), and R_D is the RoPE rotation at position D (split-half pairing:
plane p pairs local dims (p, p+64), theta_p = D / 10000^(p/64)). This uses the
relative-position identity (R_i q).(R_j k) = (R_D q).k, so only D matters.

Writes ../qk28_scores_normU.png and caches the score matrix in
cache/qk28_scores_normU.npz.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

N_HEADS, D_HEAD, N_CTX, ROPE_BASE = 6, 128, 512, 10000
CACHE = HERE / "cache" / "qk28_scores_normU.npz"


def compute() -> tuple[np.ndarray, np.ndarray]:
    """Return (alive k ids, scores (n_alive, N_CTX))."""
    from load import PILE_4L, ParameterComponents

    pc = ParameterComponents(PILE_4L / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")
    U_q = pc.components["h.0.attn.q_proj"].U.float().numpy()  # (512, 768)
    U_k = pc.components["h.0.attn.k_proj"].U.float().numpy()

    mean_ci = np.load(ROOT / "pile-qk-comps" / "hide" / "cache" / "mean_ci.npz")
    k_ids = np.where(mean_ci["h.0.attn.k_proj"] > 1e-6)[0]

    q = U_q[28] / np.linalg.norm(U_q[28])
    K = U_k[k_ids] / np.linalg.norm(U_k[k_ids], axis=1, keepdims=True)
    q = q.reshape(N_HEADS, D_HEAD)
    K = K.reshape(-1, N_HEADS, D_HEAD)

    # RoPE: theta[D, p] = D / base^(p/64), planes pair local dims (p, p+64)
    half = D_HEAD // 2
    freq = ROPE_BASE ** (np.arange(half) / half)
    theta = np.arange(N_CTX)[:, None] / freq[None, :]           # (512, 64)
    cos = np.tile(np.cos(theta), 2)                             # (512, 128)
    sin = np.tile(np.sin(theta), 2)

    q_rh = np.concatenate([-q[:, half:], q[:, :half]], axis=1)  # rotate_half
    q_rot = q[None] * cos[:, None] + q_rh[None] * sin[:, None]  # (512, 6, 128)

    scores = np.einsum("dhi,chi->cd", q_rot, K) / np.sqrt(D_HEAD)
    return k_ids, scores.astype(np.float32)


if CACHE.exists():
    d = np.load(CACHE)
    k_ids, scores = d["k_ids"], d["scores"]
else:
    k_ids, scores = compute()
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, k_ids=k_ids, scores=scores)

n = len(k_ids)
vmax = np.abs(scores).max()
print(f"{n} alive k components; max |score| = {vmax:.3f} "
      f"(unit-norm bound 1/sqrt(128) per head ~ 0.088, x6 heads if aligned)")

fig, ax = plt.subplots(figsize=(12, 15))
im = ax.imshow(scores, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               extent=(-0.5, N_CTX - 0.5, n - 0.5, -0.5), interpolation="nearest")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(k_ids, fontsize=4.5)
ax.set_xlabel("distance D = query pos $-$ key pos (tokens)")
ax.set_ylabel("alive k component (id order)")
ax.set_title("Pre-softmax score  $\\sum_h (R_D \\hat q_h)\\cdot\\hat k_{c,h}/\\sqrt{128}$"
             " between q:28 and each alive L0 k component\n"
             "(unit-normalized output vectors $U$, no data)", fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("score")
fig.tight_layout()
out = HERE.parent / "qk28_scores_normU.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
