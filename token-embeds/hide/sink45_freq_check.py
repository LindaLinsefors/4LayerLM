"""Does token frequency live in the sink-seed-45 model's (untied) embedding
and/or unembedding?

Mirrors the pile_4l methodology (token-embeds/pca.py + freq_direction.py):
counts over the 4,000 cached Pile rows; frequent = count >= 7 (missing tokens
thereby excluded); PCA on the mean-centered frequent rows of each matrix, sign
so the frequent-mean projection is >= 0; per-PC Spearman rho(proj, log count);
ridge regression of log count on the raw frequent rows, lambda by 5-fold CV,
all reported correlations cross-validated. Unfrequent (non-missing) tokens are
shown as out-of-sample points in the figure.

Output: token-embeds/sink45_freq.png + printed numbers.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_sink

rng = np.random.default_rng(0)

model, _ = load_sink(45)
mats = {
    "embedding (wte)": model.wte.weight.detach().float().numpy(),
    "unembedding (lm_head)": model.lm_head.weight.detach().float().numpy(),
}
del model

rows = torch.stack(torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                              map_location="cpu", weights_only=True))
counts = np.bincount(rows.flatten().numpy(), minlength=len(next(iter(mats.values()))))
frequent = counts >= 7
missing = np.load(ROOT / "token-embeds/hide/missing_tokens.npy")
unfrequent = ~frequent
unfrequent[missing] = False
y = np.log(counts[frequent].astype(np.float64))
print(f"frequent {frequent.sum()}, unfrequent (non-missing) {unfrequent.sum()}, "
      f"missing {len(missing)}")


def cv_ridge(X, y, k=5, lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """5-fold CV ridge of y on X (raw rows, fold-wise centered). Returns
    (best lambda, CV Spearman, CV Pearson, full-data direction w)."""
    n = len(y)
    folds = rng.permutation(n) % k
    best = None
    for lam in lambdas:
        pred = np.empty(n)
        for f in range(k):
            tr, te = folds != f, folds == f
            Xm, ym = X[tr].mean(0), y[tr].mean()
            Xc, yc = X[tr] - Xm, y[tr] - ym
            w = np.linalg.solve(Xc.T @ Xc + lam * n * np.eye(X.shape[1]), Xc.T @ yc)
            pred[te] = (X[te] - Xm) @ w + ym
        sp = spearmanr(pred, y).statistic
        pe = np.corrcoef(pred, y)[0, 1]
        if best is None or sp > best[1]:
            best = (lam, sp, pe)
    lam = best[0]
    Xc = X - X.mean(0)
    w = np.linalg.solve(Xc.T @ Xc + lam * n * np.eye(X.shape[1]),
                        Xc.T @ (y - y.mean()))
    return best[0], best[1], best[2], w / np.linalg.norm(w)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
results = {}
for row, (name, M) in enumerate(mats.items()):
    Xf = M[frequent]
    Xc = Xf - Xf.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    sign = np.where((Xf @ Vt.T).mean(0) >= 0, 1.0, -1.0)
    Vt = Vt * sign[:, None]

    proj = Xf @ Vt.T  # (n_freq, d)
    rhos = np.array([spearmanr(proj[:, i], y).statistic for i in range(20)])
    var_share = S**2 / (S**2).sum()
    lam, cv_sp, cv_pe, w = cv_ridge(Xf, y)
    results[name] = (rhos, var_share, lam, cv_sp, cv_pe, w, Vt)

    print(f"\n=== {name} ===")
    print(f"PC1: var share {var_share[0]:.3f}, Spearman(proj, log count) = {rhos[0]:+.3f}")
    top = np.argsort(-np.abs(rhos))[:5]
    print("top-|rho| PCs:", ", ".join(
        f"PC{i+1} {rhos[i]:+.3f} (var {var_share[i]:.3f})" for i in top))
    print(f"ridge (lambda {lam:g}): CV Spearman {cv_sp:+.3f}, CV Pearson {cv_pe:+.3f}")
    print(f"cos(w_ridge, PC1) = {abs(w @ Vt[0]):.3f}")

    # scatter: PC1 projection vs count (frequent + unfrequent out-of-sample)
    ax = axes[row, 0]
    pu = M[unfrequent] @ Vt[0]
    ax.scatter(counts[frequent], proj[:, 0], s=2, alpha=0.25, label="frequent")
    ax.scatter(np.maximum(counts[unfrequent], 0.5), pu, s=2, alpha=0.25,
               color="tab:orange", label="unfrequent")
    ax.set_xscale("log")
    ax.set_xlabel("token count (log)")
    ax.set_ylabel("PC1 projection")
    rho_all = spearmanr(np.concatenate([proj[:, 0], pu]),
                        np.concatenate([counts[frequent], counts[unfrequent]])).statistic
    ax.set_title(f"{name}: PC1 vs count — Spearman {rhos[0]:+.2f} (frequent), "
                 f"{rho_all:+.2f} (incl. unfrequent)")
    ax.legend(markerscale=4)

    # scatter: CV-style ridge prediction vs count (frequent, refit full = in-sample
    # shape only; the printed CV numbers are the honest ones)
    ax = axes[row, 1]
    pf = (Xf - Xf.mean(0)) @ w
    ax.scatter(counts[frequent], pf, s=2, alpha=0.25)
    ax.set_xscale("log")
    ax.set_xlabel("token count (log)")
    ax.set_ylabel("ridge-direction projection")
    ax.set_title(f"{name}: best linear direction — CV Spearman {cv_sp:+.2f}")

fig.suptitle("sink seed 45 (t-87f91319, untied): token frequency in embedding vs unembedding\n"
             "(PCs from frequent tokens, count ≥ 7 over 2.05M cached Pile tokens; "
             "ridge = 5-fold-CV best linear direction)")
fig.tight_layout()
out = HERE.parent / "sink45_freq.png"
fig.savefig(out, dpi=140)
print(f"\nsaved {out}")

# cross-matrix: is it the same direction?
(_, _, _, _, _, w_e, Vt_e) = results["embedding (wte)"]
(_, _, _, _, _, w_u, Vt_u) = results["unembedding (lm_head)"]
print(f"cos(PC1_emb, PC1_unemb) = {abs(Vt_e[0] @ Vt_u[0]):.3f}")
print(f"cos(w_ridge_emb, w_ridge_unemb) = {abs(w_e @ w_u):.3f}")
