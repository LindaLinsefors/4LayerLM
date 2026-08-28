"""The embedding-space direction that best predicts corpus token frequency.

For each model, fit ridge regression of y = log(count) on the raw embedding
rows of the frequent/alive tokens (same fit set as the PCA in pca.py):

    w = argmin_w  ||X~ w - y~||^2 + lam ||w||^2      (X~, y~ centered)

The unit vector w / ||w|| is *the* frequency direction: among all directions,
projecting onto it maximizes the (in-sample) Pearson correlation with log
count -- the multiple correlation R. lam is chosen by 5-fold CV on held-out
Spearman; reported correlations are cross-validated (out-of-fold predictions),
so they are not inflated by the d free parameters.

Outputs:
  freq_direction_{pile_4l,simple_2l}.npy -- unit direction (float32, d)
  freq_direction.png -- per model: count-vs-projection scatter (cf. freq.py's
    PC1 panel) + decomposition of w over the alive-PCA basis + the same
    decomposition weighted by the PC variance (|w . PC_k| * var_k -- how much
    each PC actually contributes to the projection's spread), full spectrum
    and zoomed to the first 50 PCs
Printed: CV Spearman/Pearson, PC1 comparison, cos(w, PC1), cos(w, mean emb).

Token sets (CLAUDE.md): fit on pile_4l frequent (count >= 7) / simple_2l
alive (freq >= 10); scatters show all tokens, pile colored by category.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent.parent
HERE = Path(__file__).parent

pile_emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
simple_emb = torch.load(
    ROOT / "prev_paper/models/simplestories_2layer/target_model_gf6rbga0/model_step_99999.pt",
    map_location="cpu", weights_only=True,
)["wte.weight"].float().numpy()

pile_rows = torch.stack(torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
))
pile_counts = np.bincount(pile_rows.flatten().numpy(), minlength=len(pile_emb))

table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
simple_counts = np.zeros(len(simple_emb), dtype=np.int64)
simple_counts[table["id"].to_numpy()] = table["freq"].to_numpy()

missing_ids = np.load(HERE / "missing_tokens.npy")

MODELS = [
    ("pile_4l", pile_emb, pile_counts, 7),
    ("simple_2l", simple_emb, simple_counts, 10),
]
CAT_COLORS = {"frequent": "#4e79a7", "unfrequent": "#e0793d", "missing": "#a1443a",
              "alive": "#4e79a7", "dead": "#e0793d"}
LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def ridge(X, y, lam):
    """Ridge solution on already-centered X, y."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


rng = np.random.default_rng(0)
fig, axes = plt.subplots(4, 2, figsize=(12, 12), height_ratios=[3, 1.6, 1.6, 1.6])

for (ax, pc_ax, var_ax, zoom_ax), (name, emb, counts, thr) in zip(axes.T, MODELS):
    alive = counts >= thr
    X, y = emb[alive], np.log(counts[alive].astype(float))
    Xc, yc = X - X.mean(0), y - y.mean()

    # 5-fold CV: pick lam by held-out Spearman, keep out-of-fold predictions
    folds = rng.permutation(len(y)) % 5
    best = None
    for lam in LAMBDAS:
        oof = np.empty(len(y))
        for k in range(5):
            tr = folds != k
            oof[~tr] = Xc[~tr] @ ridge(Xc[tr], yc[tr], lam)
        rho = spearmanr(oof, yc).statistic
        if best is None or rho > best[1]:
            best = (lam, rho, oof)
    lam, cv_rho, oof = best
    cv_r = pearsonr(oof, yc).statistic

    w = ridge(Xc, yc, lam)
    w /= np.linalg.norm(w)
    np.save(HERE / f"freq_direction_{name}.npy", w.astype(np.float32))

    # comparisons on the fit set: PC1, norm, mean-embedding direction
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = vt[0] if (X @ vt[0]).mean() >= 0 else -vt[0]
    in_rho = spearmanr(X @ w, y).statistic
    in_r = pearsonr(X @ w, y).statistic
    pc1_rho = spearmanr(X @ pc1, y).statistic
    mean_dir = emb.mean(0) / np.linalg.norm(emb.mean(0))
    print(f"{name}: fit on {alive.sum():,} tokens, lam={lam:g}")
    print(f"  freq direction: Spearman {cv_rho:+.3f} CV / {in_rho:+.3f} in-sample,"
          f" Pearson {cv_r:+.3f} CV / {in_r:+.3f} in-sample (= multiple R)")
    print(f"  PC1 for comparison: Spearman {pc1_rho:+.3f}")
    print(f"  cos(w, PC1) = {w @ pc1:+.3f},  cos(w, mean emb) = {w @ mean_dir:+.3f}")

    # scatter: count vs projection onto w, all tokens (sign: rho > 0 already)
    proj = emb @ w
    x = np.where(
        counts > 0,
        counts * np.exp(rng.uniform(-0.08, 0.08, len(counts))),
        rng.uniform(-0.35, 0.35, len(counts)),
    )
    if name == "pile_4l":
        missing = np.zeros(len(emb), dtype=bool)
        missing[missing_ids] = True
        groups = [("frequent", alive), ("unfrequent", ~alive & ~missing),
                  ("missing", missing)]
    else:
        groups = [("alive", alive), ("dead", ~alive)]
    for label, sel in groups:
        ax.scatter(x[sel], proj[sel], s=3, alpha=0.25, label=label,
                   color=CAT_COLORS[label], edgecolors="none", rasterized=True)
    ax.legend(fontsize=8, frameon=False, markerscale=3, loc="upper right")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("corpus token count (symlog; 0 = never seen)")
    ax.set_title(f"{name}:  projection onto the frequency direction", fontsize=11)
    ax.annotate(
        f"CV Spearman ρ = {cv_rho:+.3f}   (PC1: {pc1_rho:+.3f})\n"
        f"CV Pearson r(log count) = {cv_r:+.3f}",
        xy=(0.02, 0.02), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=8.5, color="#333333",
    )
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # where w lives in the alive-PCA basis
    coeffs = vt @ w
    pc_ax.plot(np.arange(1, len(coeffs) + 1), np.abs(coeffs),
               color="#4e79a7", linewidth=0.8)
    pc_ax.set_xlabel("alive-PCA component index")
    pc_ax.set_title(f"|w · PC_k|   (top 3: "
                    + ", ".join(f"{c:+.2f}" for c in coeffs[:3]) + ")", fontsize=10)
    pc_ax.grid(alpha=0.25, linewidth=0.5)
    pc_ax.spines[["top", "right"]].set_visible(False)

    # same coefficients weighted by PC variance: each PC's contribution to the
    # spread of the projection (var(X w) = sum_k (w . PC_k)^2 var_k)
    var = s**2 / alive.sum()
    var_ax.plot(np.arange(1, len(coeffs) + 1), np.abs(coeffs) * var,
                color="#4e79a7", linewidth=0.8)
    var_ax.set_xlabel("alive-PCA component index")
    var_ax.set_title("|w · PC_k| · var(PC_k)", fontsize=10)
    var_ax.grid(alpha=0.25, linewidth=0.5)
    var_ax.spines[["top", "right"]].set_visible(False)

    zoom_ax.plot(np.arange(1, 51), np.abs(coeffs[:50]) * var[:50],
                 color="#4e79a7", linewidth=1.0, marker=".", markersize=4)
    zoom_ax.set_xlabel("alive-PCA component index")
    zoom_ax.set_title("|w · PC_k| · var(PC_k),  first 50 PCs", fontsize=10)
    zoom_ax.grid(alpha=0.25, linewidth=0.5)
    zoom_ax.spines[["top", "right"]].set_visible(False)

axes[0, 0].set_ylabel("embedding · w   (unit w)")
axes[1, 0].set_ylabel("|w · PC_k|")
axes[2, 0].set_ylabel("|w · PC_k| · var(PC_k)")
axes[3, 0].set_ylabel("|w · PC_k| · var(PC_k)")
fig.suptitle("Best linear frequency direction in embedding space", y=0.995)
fig.tight_layout()
fig.savefig(HERE.parent / "freq_direction.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE / 'freq_direction.png'}")
