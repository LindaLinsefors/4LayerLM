"""The embedding-space direction that best predicts a token's frequency of
appearing right after <|endoftext|> (= opening a document).

Fit set: the top 50% of the vocab by after-EOS count (ties at count 0 are
excluded — log is undefined there and their ranking is arbitrary; if fewer
than 50% of tokens have nonzero counts, the fit set is all nonzero ones and
the script says so).

Method (same as token-embeds/hide/freq_direction.py): ridge regression of
y = log(after-EOS count) on the centered raw embedding rows,

    w = argmin_w ||X~ w - y~||^2 + lam ||w||^2,

lam picked by 5-fold CV on held-out Spearman; reported correlations are
cross-validated. Secondary fit: same but y residualized on log(total count)
over the same shards — the document-opener direction *beyond* general
frequency.

Outputs:
  endoftext-pos0/after_eos_direction.png
  endoftext-pos0/hide/after_eos_direction.npy — unit w (float32, 768)
Printed: CV correlations, top openers, cosines vs the general frequency
direction and the fit-set PC1.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def ridge(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def cv_fit(Xc, yc, rng):
    """Pick lam by 5-fold CV Spearman; return (w_unit, lam, cv_rho, cv_r)."""
    folds = rng.permutation(len(yc)) % 5
    best = None
    for lam in LAMBDAS:
        oof = np.empty(len(yc))
        for k in range(5):
            tr = folds != k
            oof[~tr] = Xc[~tr] @ ridge(Xc[tr], yc[tr], lam)
        rho = spearmanr(oof, yc).statistic
        if best is None or rho > best[1]:
            best = (lam, rho, oof)
    lam, cv_rho, oof = best
    w = ridge(Xc, yc, lam)
    return w / np.linalg.norm(w), lam, cv_rho, pearsonr(oof, yc).statistic


emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
d = np.load(HERE / "cache/after_eos_counts.npz")
eos_counts, total_counts = d["after_eos_counts"], d["total_counts"]
tok = load_tokenizer("pile_4l")

nonzero = np.count_nonzero(eos_counts)
half = len(emb) // 2
n_fit = min(half, nonzero)
fit_idx = np.argsort(eos_counts)[::-1][:n_fit]  # top tokens by after-EOS count
print(f"{d['n_after_eos']:,} after-EOS samples over {d['n_tokens']:,} tokens; "
      f"{nonzero:,} distinct openers ({nonzero / len(emb):.1%} of vocab)")
print(f"fit set: top {n_fit:,} tokens"
      + (f" (all nonzero — fewer than the requested 50% = {half:,})" if n_fit < half else " (= top 50%)"))
top = np.argsort(eos_counts)[::-1][:15]
print("top openers:", ", ".join(f"{tok.decode([int(i)])!r}:{eos_counts[i]:,}" for i in top))

X = emb[fit_idx]
y = np.log(eos_counts[fit_idx].astype(float))
Xc, yc = X - X.mean(0), y - y.mean()
rng = np.random.default_rng(0)

w, lam, cv_rho, cv_r = cv_fit(Xc, yc, rng)
np.save(HERE / "after_eos_direction.npy", w.astype(np.float32))

# baselines & comparisons
_, s, vt = np.linalg.svd(Xc, full_matrices=False)
pc1 = vt[0] if (X @ vt[0]).mean() >= 0 else -vt[0]
w_freq = np.load(ROOT / "token-embeds/hide/freq_direction_pile_4l.npy")
log_tot = np.log(total_counts[fit_idx].astype(float))  # every opener also occurs
print(f"lam={lam:g};  w_eos: CV Spearman {cv_rho:+.3f}, CV Pearson(log count) {cv_r:+.3f}")
print(f"  PC1 baseline Spearman {spearmanr(X @ pc1, y).statistic:+.3f}; "
      f"general-freq-direction baseline {spearmanr(X @ w_freq, y).statistic:+.3f}")
print(f"  cos(w_eos, w_freq) = {w @ w_freq:+.3f},  cos(w_eos, fit-set PC1) = {w @ pc1:+.3f}")
print(f"  Spearman(log after-EOS count, log total count) on fit set: "
      f"{spearmanr(y, log_tot).statistic:+.3f}")

# secondary: opener-specificity — residualize y on log total count, refit
A = np.stack([np.ones(len(y)), log_tot], axis=1)
y_res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
w2, lam2, cv_rho2, cv_r2 = cv_fit(Xc, y_res - y_res.mean(), rng)
print(f"residual fit (after-EOS freq beyond total freq): lam={lam2:g}, "
      f"CV Spearman {cv_rho2:+.3f}, CV Pearson {cv_r2:+.3f}, "
      f"cos(w_res, w_eos) = {w2 @ w:+.3f}, cos(w_res, w_freq) = {w2 @ w_freq:+.3f}")

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(ax_sc, ax_cmp), (ax_pc, ax_var) = axes
rj = np.random.default_rng(1)

proj = emb @ w
in_fit = np.zeros(len(emb), dtype=bool)
in_fit[fit_idx] = True
xj = np.where(eos_counts > 0,
              eos_counts * np.exp(rj.uniform(-0.08, 0.08, len(emb))),
              rj.uniform(-0.35, 0.35, len(emb)))
for label, sel, color in [
    (f"fit set (top {n_fit:,})", in_fit, "#4e79a7"),
    ("other openers", ~in_fit & (eos_counts > 0), "#e0793d"),
    ("never after EOS", eos_counts == 0, "#b5b5b5"),
]:
    ax_sc.scatter(xj[sel], proj[sel], s=3, alpha=0.25, label=label,
                  color=color, edgecolors="none", rasterized=True)
ax_sc.set_xscale("symlog", linthresh=1)
ax_sc.set_xlabel("after-EOS count (symlog; 0 = never seen after EOS)")
ax_sc.set_ylabel("embedding · w_eos")
ax_sc.set_title("projection onto the after-EOS frequency direction", fontsize=11)
ax_sc.legend(fontsize=8, frameon=False, markerscale=3, loc="upper left")
ax_sc.annotate(f"CV Spearman ρ = {cv_rho:+.3f}\nCV Pearson r(log count) = {cv_r:+.3f}",
               xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
               fontsize=8.5, color="#333333")

sel = eos_counts > 0
ax_cmp.scatter(total_counts[sel] * np.exp(rj.uniform(-0.05, 0.05, sel.sum())),
               eos_counts[sel] * np.exp(rj.uniform(-0.05, 0.05, sel.sum())),
               s=3, alpha=0.25, color="#4e79a7", edgecolors="none", rasterized=True)
ax_cmp.set_xscale("log"); ax_cmp.set_yscale("log")
ax_cmp.set_xlabel("total count (same shards)")
ax_cmp.set_ylabel("after-EOS count")
ax_cmp.set_title(f"after-EOS vs total frequency (openers only; Spearman "
                 f"{spearmanr(np.log(total_counts[sel]), np.log(eos_counts[sel])).statistic:+.2f})",
                 fontsize=11)

coeffs = vt @ w
var = s ** 2 / n_fit
ax_pc.plot(np.arange(1, len(coeffs) + 1), np.abs(coeffs), color="#4e79a7", lw=0.8)
ax_pc.set_xlabel("fit-set PCA component index")
ax_pc.set_title(f"|w_eos · PC_k|   (cos(w_eos, w_freq) = {w @ w_freq:+.2f})", fontsize=11)
ax_var.plot(np.arange(1, len(coeffs) + 1), np.abs(coeffs) * var, color="#4e79a7", lw=0.8)
ax_var.set_xlabel("fit-set PCA component index")
ax_var.set_title("|w_eos · PC_k| · var(PC_k)", fontsize=11)
for ax in (ax_sc, ax_cmp, ax_pc, ax_var):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("pile_4l — embedding direction predicting after-<|endoftext|> (document-opener) frequency",
             y=0.995)
fig.tight_layout()
fig.savefig(HERE.parent / "after_eos_direction.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'after_eos_direction.png'}")
