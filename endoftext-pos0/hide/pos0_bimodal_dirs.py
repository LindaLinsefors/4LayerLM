"""Projection pursuit: which directions make the L0 position-0 value-mixture
distribution maximally bimodal?

Search: minimize the Pearson kurtosis of w.y over unit directions w (kurtosis
2-point +-c -> 1, Gaussian -> 3; kurtosis minima are the classic projection-
pursuit objective for clustered/bimodal projections). Performed in the
whitened top-K PCA space of the pos-0 samples, batched multi-restart Adam,
then 2 more directions by deflation (orthogonal in whitened space).

Overfitting control: optimize on the even rows, report ALL statistics on the
held-out odd rows; a matched multivariate-Gaussian null (same top-K covariance)
is pushed through the same pipeline to show what kurtosis pure overfitting
produces.

Bimodality measures reported (held-out): Pearson kurtosis, Sarle's bimodality
coefficient BC = (skew^2 + 1)/kurt (> 5/9 suggests bimodality), and from a 1-D
two-Gaussian EM fit: Ashman's D = |mu1 - mu2| / sqrt((s1^2 + s2^2)/2) (> 2 =
well-separated modes) with the mixture weights.

Position relevance per direction: var-ratio pos0/bulk, linear AUC, folded AUC,
|cos| with the mean-difference direction and the top-3 variance-ratio
directions (pos0_variance_signal.npz). Also: 2-D scatter in the top-2 bimodal
directions (a grid of clusters = multiple binary dimensions), and the most
common tokens in each lobe of direction 1.

Output: endoftext-pos0/pos0_bimodal_dirs.png + printed tables.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l, load_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 4000            # all cached rows; ~2000 train / ~2000 held-out pos-0 samples
BATCH_SIZE = 16
EOS_ID = 0
T = 512
N_BULK = 4
BULK_MIN = 128
K = 100                  # PCA dims searched
N_RESTART = 24
N_STEPS = 600
N_DIRS = 60              # deflation depth (kurt-vs-rank curve); top 3 shown in detail

# ------------------------------------------------------------------- capture
rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]
rng = np.random.default_rng(0)
bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))

model, _, _ = load_pile_4l()
model = model.to(DEVICE)
caps: dict[int, torch.Tensor] = {}
hook = model.h[0].attn.o_proj.register_forward_pre_hook(
    lambda _m, inp: caps.__setitem__(0, inp[0]))

y0, y0_tok, yb = [], [], []
with torch.no_grad():
    for start in range(0, N_ROWS, BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        tok = batch.cpu().numpy()
        for bi in range(len(tok)):
            ri = start + bi
            y = caps[0][bi].float().cpu().numpy()
            if tok[bi, 0] != EOS_ID:
                y0.append(y[0])
                y0_tok.append(tok[bi, 0])
            for p in bulk_pos[ri]:
                if tok[bi, p] != EOS_ID:
                    yb.append(y[p])
hook.remove()
del model
Y0 = np.array(y0, dtype=np.float64)
Yb = np.array(yb, dtype=np.float64)
y0_tok = np.array(y0_tok)
tr = np.arange(len(Y0)) % 2 == 0
te = ~tr
print(f"pos-0 samples: {len(Y0)} (train {tr.sum()}, test {te.sum()}); bulk {len(Yb)}")


# --------------------------------------------------------- projection pursuit
def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def pursue(Z: np.ndarray, n_dirs: int, seed: int) -> np.ndarray:
    """Minimize kurtosis of Z @ w over unit w (whitened space), n_dirs by
    deflation; batched restarts. Returns (dim, n_dirs)."""
    g = torch.Generator().manual_seed(seed)
    Zt = torch.tensor(Z)
    basis = torch.eye(Z.shape[1], dtype=torch.float64)
    dirs = []
    for _ in range(n_dirs):
        Zp = Zt @ basis                            # restrict to remaining subspace
        w = torch.randn(Zp.shape[1], N_RESTART, generator=g, dtype=torch.float64)
        w = torch.nn.Parameter(w)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(N_STEPS):
            opt.zero_grad()
            wn = w / w.norm(dim=0, keepdim=True)
            x = Zp @ wn                            # (n, R)
            xc = x - x.mean(0)
            loss = ((xc ** 4).mean(0) / (xc ** 2).mean(0) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            wn = w / w.norm(dim=0, keepdim=True)
            x = Zp @ wn
            xc = x - x.mean(0)
            ks = (xc ** 4).mean(0) / (xc ** 2).mean(0) ** 2
            best = basis @ wn[:, ks.argmin()]
        dirs.append(best / best.norm())
        # deflate: orthocomplement of found directions
        D = torch.stack(dirs, 1)
        q, _ = torch.linalg.qr(torch.eye(Z.shape[1], dtype=torch.float64)
                               - D @ D.T)
        basis = q[:, : Z.shape[1] - len(dirs)]
    return torch.stack(dirs, 1).numpy()


mu_tr = Y0[tr].mean(0)
Xc = Y0[tr] - mu_tr
U_, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Vt[:K].T                                       # (768, K)
scale = S_[:K] / np.sqrt(tr.sum() - 1)
to_white = lambda Y: ((Y - mu_tr) @ P) / scale

Ztr = to_white(Y0[tr])
Wwhite = pursue(Ztr, N_DIRS, seed=0)               # (K, N_DIRS)

# matched Gaussian null through the same pipeline
Zn = rng.standard_normal(Ztr.shape)
Wnull = pursue(Zn, 1, seed=1)
null_te = rng.standard_normal((te.sum(), K)) @ Wnull[:, 0]
print(f"\nGaussian null (same n, K, optimizer): train-optimized dir has held-out "
      f"kurtosis {kurt(null_te):.2f} (Gaussian = 3; anything near this on real "
      f"data is overfitting, far below it is real structure)")


# --------------------------------------------------------------- evaluation
def em2(x, iters=200):
    """1-D two-Gaussian EM; returns (means, stds, weights) sorted by mean."""
    m = np.percentile(x, [25, 75]).astype(float)
    s = np.array([x.std() / 2] * 2)
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        logp = -0.5 * ((x[:, None] - m) / s) ** 2 - np.log(s) + np.log(pi)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)
        n = r.sum(0)
        m = (r * x[:, None]).sum(0) / n
        s = np.sqrt((r * (x[:, None] - m) ** 2).sum(0) / n).clip(1e-9)
        pi = n / len(x)
    o = np.argsort(m)
    return m[o], s[o], pi[o]


def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


cache = np.load(HERE / "cache" / "pos0_variance_signal.npz")
Wvar = cache["L0_W"][:, :3]
dmean = Y0.mean(0) - Yb.mean(0)
dmean /= np.linalg.norm(dmean)

# directions mapped to y-space for projecting any sample set
proj_dir = lambda Y, i: to_white(Y) @ Wwhite[:, i]
wy = P @ (Wwhite / scale[:, None])                 # unnormalized y-space normals
wy /= np.linalg.norm(wy, axis=0)

kurt_te = np.array([kurt(proj_dir(Y0[te], i)) for i in range(N_DIRS)])
print(f"\nheld-out kurtosis by direction rank: "
      + " ".join(f"{k:.2f}" for k in kurt_te))
# reference ranks with intermediate bimodality, by kurtosis thresholds
REF_RANKS = []
for t in (2.0, 2.4, 2.8):
    cand = np.where(kurt_te >= t)[0]
    cand = [c for c in cand if c not in REF_RANKS and c > 2]
    if cand:
        REF_RANKS.append(int(cand[0]))
for frac in (0.33, 0.66, 1.0):          # fallback: spread over the curve
    if len(REF_RANKS) >= 3:
        break
    r = min(int(frac * (N_DIRS - 1)), N_DIRS - 1)
    if r not in REF_RANKS and r > 2:
        REF_RANKS.append(r)
REF_RANKS = sorted(REF_RANKS)[:3]

N_SHOW = 3
print(f"\nL0 pos-0 bimodal directions (all stats on held-out half, n={te.sum()}):")
print("dir | kurt | BC(>0.56) | AshmanD | weights | var-ratio | linAUC | foldAUC | cos(dmean) | max|cos(vardir)|")
for i in list(range(N_SHOW)) + REF_RANKS:
    x = proj_dir(Y0[te], i)
    xb = proj_dir(Yb, i)
    xc = x - x.mean()
    kk = kurt(x)
    skew = float((xc ** 3).mean() / (xc ** 2).mean() ** 1.5)
    bc = (skew ** 2 + 1) / kk
    m, s, pi = em2(x)
    D = abs(m[1] - m[0]) / np.sqrt((s[0] ** 2 + s[1] ** 2) / 2)
    vr = x.var() / xb.var()
    la = auc(xb, x)
    fa = auc(np.abs(xb - xb.mean()), np.abs(x - xb.mean()))
    cd = abs(float(wy[:, i] @ dmean))
    cv = np.abs(wy[:, i] @ Wvar).max()
    print(f"  {i} | {kk:4.2f} | {bc:8.2f} | {D:7.2f} | {pi[0]:.2f}/{pi[1]:.2f} "
          f"| {vr:9.1f} | {la:6.3f} | {fa:7.3f} | {cd:10.2f} | {cv:8.2f}")

# token lobes of direction 0 (full sample, split at the EM decision boundary)
x_all = proj_dir(Y0, 0)
m, s, pi = em2(x_all)
grid = np.linspace(m[0], m[1], 1001)
ll = (-0.5 * ((grid[:, None] - m) / s) ** 2 - np.log(s) + np.log(pi))
bnd = grid[np.argmin(np.abs(ll[:, 0] - ll[:, 1]))]
tokz = load_tokenizer("pile_4l")
print(f"\ndirection 0 lobes (EM boundary {bnd:.2f}): "
      f"low {np.sum(x_all < bnd)} / high {np.sum(x_all >= bnd)} tokens")
for name, mask in [("low", x_all < bnd), ("high", x_all >= bnd)]:
    ids, cnt = np.unique(y0_tok[mask], return_counts=True)
    o = np.argsort(-cnt)[:15]
    print(f"  {name} lobe top tokens: " + ", ".join(
        f"{tokz.decode([int(ids[j])])!r}x{cnt[j]}" for j in o))

# ---------------------------------------------------------------------- plot
def hist_panel(ax, i, title):
    x = proj_dir(Y0[te], i)
    xb = proj_dir(Yb, i)
    lo, hi = np.percentile(x, [0.2, 99.8])
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 45)
    d0, _, _ = ax.hist(x, bins=bins, density=True, color="tab:orange", alpha=0.75,
                       label=f"pos 0, held-out (n={te.sum()})")
    ax.hist(xb, bins=bins, density=True, color="tab:blue", alpha=0.45,
            label="bulk (peak clipped)")
    ax.set_ylim(0, 1.25 * d0.max())
    ax.set_title(f"{title}: held-out kurt {kurt(x):.2f}", fontsize=11)
    ax.set_xlabel("w·y")
    ax.legend(fontsize=8, frameon=False)


fig, axes = plt.subplots(2, 4, figsize=(18, 9.2))
for i in range(3):
    hist_panel(axes[0, i], i, f"bimodal dir {i}")

ax = axes[0, 3]
x1, x2 = proj_dir(Y0[te], 0), proj_dir(Y0[te], 1)
ax.plot(x1, x2, ".", ms=3.5, alpha=0.5, color="tab:orange")
ax.set_xlabel("bimodal dir 0")
ax.set_ylabel("bimodal dir 1")
ax.set_title("pos-0 samples in the top-2 bimodal directions\n"
             "(grid of clusters = independent binary dims)", fontsize=11)

ax = axes[1, 0]
ax.plot(np.arange(N_DIRS), kurt_te, "o-", ms=4, color="tab:orange",
        label="held-out kurtosis")
ax.axhline(3.0, color="k", ls=":", lw=1, label="Gaussian (3)")
ax.axhline(kurt(null_te), color="tab:gray", ls="--", lw=1,
           label=f"matched-Gaussian null ({kurt(null_te):.2f})")
for r in REF_RANKS:
    ax.plot(r, kurt_te[r], "s", ms=8, mfc="none", mec="tab:red")
ax.set_xlabel("direction rank (deflation order)")
ax.set_ylabel("held-out kurtosis of pos-0 projection")
ax.set_title("bimodality vs rank (lower = more bimodal;\n"
             "red squares = reference dirs shown right)", fontsize=11)
ax.legend(fontsize=8, frameon=False)

for j, r in enumerate(REF_RANKS[:3]):
    hist_panel(axes[1, j + 1], r, f"reference dir {r}")
for j in range(len(REF_RANKS), 3):
    axes[1, j + 1].axis("off")

for ax in axes.flat:
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — projection pursuit for maximal position-0 bimodality, "
             f"layer-0 value mixture (kurtosis minimization in top-{K} PC space; "
             f"optimized on half the rows, shown on the other half; "
             f"Gaussian-null held-out kurtosis {kurt(null_te):.2f})", fontsize=11)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_bimodal_dirs.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {HERE.parent / 'pos0_bimodal_dirs.png'}")
