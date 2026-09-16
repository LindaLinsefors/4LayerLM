"""The covariance direction v_cov for after-EOS (document-opener) frequency,
and where it sits in the embedding PCA basis.

v_cov = normalize(Xc^T yc) over the fit set (all 8,243 openers), with
X = raw opener embeddings, y = log(after-EOS count), both centered on the fit
set — the direction whose projection has maximal linear covariance with y
(cf. remove-feq-dir: for *general* frequency v_cov ~= PC1, cos 0.97).

Compared against the project-standard embedding PCA basis: PCs of the
frequent-token embeddings (count >= 7 over the 4,000 cached rows,
mean-centered on that set, sign: frequent-mean projection >= 0 — identical to
token-embeds/hide/pca.py). Random-direction baseline |cos| = 1/sqrt(768).

Outputs: endoftext-pos0/after_eos_vcov.png, hide/after_eos_vcov.npy (unit vector).
Printed: in-sample correlations, top-|cos| PCs, cosines vs w_eos / PC1 /
general-frequency v_cov / w_freq.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
d = np.load(HERE / "cache/after_eos_counts.npz")
eos_counts, total_counts = d["after_eos_counts"], d["total_counts"]

# fit set: all openers (the realizable "top 50%", see report §6)
fit = eos_counts > 0
X = emb[fit]
y = np.log(eos_counts[fit].astype(float))
Xc, yc = X - X.mean(0), y - y.mean()
v = Xc.T @ yc
v /= np.linalg.norm(v)
np.save(HERE / "after_eos_vcov.npy", v.astype(np.float32))

# project-standard embedding PCA basis (frequent tokens, count >= 7, cached rows)
rows = torch.stack(torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                              map_location="cpu", weights_only=True))
counts_rows = np.bincount(rows.flatten().numpy(), minlength=len(emb))
freq = counts_rows >= 7
Xf = emb[freq] - emb[freq].mean(0)
_, s_f, vt_f = np.linalg.svd(Xf, full_matrices=False)
sign = np.where((emb[freq] @ vt_f.T).mean(0) >= 0, 1.0, -1.0)
pcs = vt_f * sign[:, None]                      # (768, 768) rows = PCs
var_f = s_f ** 2 / freq.sum()

cos = pcs @ v                                   # signed cos with each PC
baseline = 1 / np.sqrt(emb.shape[1])
order = np.argsort(np.abs(cos))[::-1]

# comparisons
w_eos = np.load(HERE / "after_eos_direction.npy")
w_freq = np.load(ROOT / "token-embeds/hide/freq_direction_pile_4l.npy")
yf = np.log(counts_rows[freq].astype(float))
v_freq = Xf.T @ (yf - yf.mean())
v_freq /= np.linalg.norm(v_freq)

print(f"fit set: {fit.sum():,} openers; in-sample Spearman(X v, y) = "
      f"{spearmanr(X @ v, y).statistic:+.3f} (ridge w_eos CV was +0.898)")
print(f"cos(v_cov_eos, w_eos)  = {v @ w_eos:+.3f}")
print(f"cos(v_cov_eos, w_freq) = {v @ w_freq:+.3f},  "
      f"cos(v_cov_eos, v_cov_freq) = {v @ v_freq:+.3f},  "
      f"cos(v_cov_eos, PC1) = {cos[0]:+.3f}")
print(f"random baseline |cos| = {baseline:.3f}")
print("top-10 |cos| PCs: "
      + ", ".join(f"PC{k + 1}:{cos[k]:+.3f}" for k in order[:10]))
for k in [1, 5, 10, 20, 50, 100]:
    m = np.sqrt(np.sum(np.sort(np.abs(cos))[::-1][:k] ** 2))
    print(f"  ||proj onto top-{k:3d} |cos| PCs|| = {m:.3f}")

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax_c, ax_cum, ax_sc = axes

ax_c.plot(np.arange(1, 769), np.abs(cos), lw=0.8, color="#4e79a7")
ax_c.axhline(baseline, color="#a1443a", ls="--", lw=1,
             label=f"random baseline 1/√768 = {baseline:.3f}")
for k in order[:3]:
    ax_c.annotate(f"PC{k + 1}", (k + 1, abs(cos[k])), fontsize=8,
                  xytext=(5, 2), textcoords="offset points")
ax_c.set_xlabel("frequent-token embedding PC index")
ax_c.set_ylabel("|cos(v_cov_eos, PC_k)|")
ax_c.set_title("alignment with each embedding PC", fontsize=11)
ax_c.legend(fontsize=8, frameon=False)

csum = np.sqrt(np.cumsum(np.sort(np.abs(cos))[::-1] ** 2))
ax_cum.plot(np.arange(1, 769), csum, lw=1.2, color="#4e79a7")
ax_cum.set_xscale("log")
ax_cum.set_xlabel("k (PCs, sorted by |cos|)")
ax_cum.set_ylabel("‖projection onto top-k PCs‖")
ax_cum.set_title("cumulative mass over best-aligned PCs", fontsize=11)
ax_cum.set_ylim(0, 1.02)

rj = np.random.default_rng(1)
proj = emb @ v
xj = np.where(eos_counts > 0,
              eos_counts * np.exp(rj.uniform(-0.08, 0.08, len(emb))),
              rj.uniform(-0.35, 0.35, len(emb)))
for label, sel, color in [
    ("openers (fit set)", fit, "#4e79a7"),
    ("never after EOS", ~fit, "#b5b5b5"),
]:
    ax_sc.scatter(xj[sel], proj[sel], s=3, alpha=0.25, label=label,
                  color=color, edgecolors="none", rasterized=True)
ax_sc.set_xscale("symlog", linthresh=1)
ax_sc.set_xlabel("after-EOS count (symlog)")
ax_sc.set_ylabel("embedding · v_cov_eos")
ax_sc.set_title(f"projection vs after-EOS count "
                f"(Spearman {spearmanr(X @ v, y).statistic:+.2f} on fit set)", fontsize=11)
ax_sc.legend(fontsize=8, frameon=False, markerscale=3, loc="upper left")

for ax in axes:
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("pile_4l — covariance direction v_cov of after-<|endoftext|> frequency "
             "vs the frequent-token embedding PCA basis", y=1.0)
fig.tight_layout()
fig.savefig(HERE.parent / "after_eos_vcov.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'after_eos_vcov.png'}")
