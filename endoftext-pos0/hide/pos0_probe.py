"""When does "I am position 0" become linearly readable from the residual stream?

For every stage of pile_4l (after embedding, after each attention, after each
MLP, after the final norm) and every early chunk position p in EARLY, fit a
linear probe separating "residual vector at position p" from "residual vector
at a bulk position (>= 128)" and evaluate it on held-out rows:

  * probe 1 (LDA):       w = (Sigma_bulk + lambda * tr(Sigma)/d * I)^{-1} (mu_p - mu_bulk),
                         lambda from a small grid, best CV AUC reported
  * probe 2 (mean diff): w = mu_p - mu_bulk  (no whitening)

Reported per (stage, p): cross-validated AUC (5 folds by row) and the held-out
separation d' = |m1 - m0| / sqrt((v1 + v0)/2) of the projected scores
(AUC = Phi(d'/sqrt2) for Gaussian classes — d' keeps resolving after AUC
saturates at 1). Chunk boundaries fall at random points inside documents, so
the *token distribution is identical at every position* — after-embedding AUC
is a chance-level control, and any later readability is genuine position
information created by attention.

EOS-token positions are excluded from both classes (mid-sequence EOS is its
own sink story); their activations are cached separately and used for the
massive-vector direction check at after-MLP-2 (pairwise cosines within the
position-0 group, and position-0 group vs EOS group).

Data: first N_ROWS cached Pile rows. Forward pass ~1 min local GPU, cached to
hide/cache/pos0_probe_acts.npz (float16, ~320 MB — gitignored); probing is
pure numpy afterwards. Output: endoftext-pos0/pos0_probe.png + printed tables.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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
N_BULK = 6            # random bulk positions (>= BULK_MIN) sampled per row
BULK_MIN = 128
N_FOLDS = 5
LAMBDAS = [3e-3, 3e-2, 3e-1]
CACHE = HERE / "cache" / "pos0_probe_acts.npz"


# ---------------------------------------------------------------- forward pass
def compute_acts() -> dict:
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)[:N_ROWS]
    rows = [r[:T] for r in rows]
    rng = np.random.default_rng(0)
    bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))
    keep = np.concatenate([np.array(EARLY), np.zeros(N_BULK, int)])  # per-row filled below

    model, _, _ = load_pile_4l()
    model = model.to(DEVICE)
    n_layer = len(model.h)
    stages = (["after embedding"]
              + [f"after {kind} {l + 1}" for l in range(n_layer) for kind in ("attention", "MLP")]
              + ["after final norm"])

    caps: dict[str, torch.Tensor] = {}
    hooks = []
    for l, block in enumerate(model.h):
        hooks.append(block.attn.register_forward_hook(
            lambda _m, _i, out, l=l: caps.__setitem__(f"attn{l}", out)))
        hooks.append(block.register_forward_hook(
            lambda _m, _i, out, l=l: caps.__setitem__(f"block{l}", out)))
    hooks.append(model.ln_f.register_forward_hook(
        lambda _m, _i, out: caps.__setitem__("ln_f", out)))

    n_keep = len(EARLY) + N_BULK
    acts = np.zeros((len(stages), N_ROWS, n_keep, 768), dtype=np.float16)
    kept_tok = np.zeros((N_ROWS, n_keep), dtype=np.int64)
    eos_acts, eos_where = [], []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
            model(batch)
            hs = [model.wte(batch)]
            for l in range(n_layer):
                hs.append(hs[-1] + caps[f"attn{l}"])
                hs.append(caps[f"block{l}"])
            hs.append(caps["ln_f"])
            tok = batch.cpu().numpy()
            for bi in range(len(tok)):
                ri = start + bi
                pos = np.concatenate([np.array(EARLY), bulk_pos[ri]])
                kept_tok[ri] = tok[bi, pos]
                for si, h in enumerate(hs):
                    acts[si, ri] = h[bi, pos].float().cpu().numpy()
                for p in np.nonzero(tok[bi] == EOS_ID)[0]:
                    if p > 0:
                        eos_where.append((ri, int(p)))
                        eos_acts.append(np.stack(
                            [h[bi, p].float().cpu().numpy() for h in hs]))
    for h in hooks:
        h.remove()
    del model
    out = dict(acts=acts, kept_tok=kept_tok, bulk_pos=bulk_pos,
               eos_acts=np.stack(eos_acts).transpose(1, 0, 2).astype(np.float16),
               eos_where=np.array(eos_where), stages=np.array(stages))
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    print(f"cached {CACHE} ({CACHE.stat().st_size / 1e6:.0f} MB)")
    return out


if CACHE.exists():
    out = dict(np.load(CACHE, allow_pickle=False))
    print(f"loaded cache {CACHE}")
else:
    out = compute_acts()
stages = [str(s) for s in out["stages"]]
acts, kept_tok = out["acts"], out["kept_tok"]
n_stages = len(stages)


# ------------------------------------------------------------------- probing
def auc(s0: np.ndarray, s1: np.ndarray) -> float:
    """P(score of class-1 sample > class-0 sample), rank-based."""
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


def dprime(s0: np.ndarray, s1: np.ndarray) -> float:
    return abs(s1.mean() - s0.mean()) / np.sqrt((s1.var() + s0.var()) / 2)


folds = np.arange(N_ROWS) % N_FOLDS
bulk_slots = slice(len(EARLY), None)
bulk_ok = kept_tok[:, bulk_slots] != EOS_ID       # (N_ROWS, N_BULK)

auc_lda = np.zeros((n_stages, len(EARLY)))
auc_md = np.zeros((n_stages, len(EARLY)))
dp_lda = np.zeros((n_stages, len(EARLY)))
for si in range(n_stages):
    A = acts[si].astype(np.float64)               # (rows, keep, 768)
    bulk = A[:, bulk_slots]                       # (rows, N_BULK, 768)
    # per-fold bulk stats (train = rows outside fold)
    fold_stats = []
    for f in range(N_FOLDS):
        tr = folds != f
        Xb = bulk[tr][bulk_ok[tr]]
        mu_b = Xb.mean(0)
        Xc = Xb - mu_b
        Sig = Xc.T @ Xc / (len(Xb) - 1)
        fold_stats.append((mu_b, Sig, np.trace(Sig) / Sig.shape[0]))
    for pi, p in enumerate(EARLY):
        ok1 = kept_tok[:, pi] != EOS_ID
        s0_by_lam = {lam: [] for lam in LAMBDAS + ["md"]}
        s1_by_lam = {lam: [] for lam in LAMBDAS + ["md"]}
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            mu_b, Sig, scale = fold_stats[f]
            mu_p = A[tr & ok1, pi].mean(0)
            d = mu_p - mu_b
            X0 = bulk[te][bulk_ok[te]]
            X1 = A[te & ok1, pi]
            for lam in LAMBDAS:
                w = np.linalg.solve(Sig + lam * scale * np.eye(768), d)
                s0_by_lam[lam].append(X0 @ w)
                s1_by_lam[lam].append(X1 @ w)
            s0_by_lam["md"].append(X0 @ d)
            s1_by_lam["md"].append(X1 @ d)
        best = max(LAMBDAS, key=lambda lam: auc(np.concatenate(s0_by_lam[lam]),
                                                np.concatenate(s1_by_lam[lam])))
        s0, s1 = np.concatenate(s0_by_lam[best]), np.concatenate(s1_by_lam[best])
        auc_lda[si, pi], dp_lda[si, pi] = auc(s0, s1), dprime(s0, s1)
        auc_md[si, pi] = auc(np.concatenate(s0_by_lam["md"]),
                             np.concatenate(s1_by_lam["md"]))

print("\nCV AUC (LDA probe, best lambda) — rows = stages, cols = position p:")
print("stage".ljust(20) + "".join(f"p={p}".rjust(7) for p in EARLY))
for si, st in enumerate(stages):
    print(st.ljust(20) + "".join(f"{auc_lda[si, pi]:7.3f}" for pi in range(len(EARLY))))
print("\nheld-out d' (LDA probe):")
print("stage".ljust(20) + "".join(f"p={p}".rjust(7) for p in EARLY))
for si, st in enumerate(stages):
    print(st.ljust(20) + "".join(f"{dp_lda[si, pi]:7.2f}" for pi in range(len(EARLY))))
print("\nCV AUC (mean-difference probe), p=0 column by stage:")
for si, st in enumerate(stages):
    print(f"  {st.ljust(20)} LDA {auc_lda[si, 0]:.3f}   mean-diff {auc_md[si, 0]:.3f}")

# ------------------------------------------- massive-vector direction checks
si_mlp2 = stages.index("after MLP 2")
p0 = acts[si_mlp2, kept_tok[:, 0] != EOS_ID, 0].astype(np.float64)
p0u = p0 / np.linalg.norm(p0, axis=1, keepdims=True)
C = p0u @ p0u.T
mean_pair = (C.sum() - len(p0u)) / (len(p0u) ** 2 - len(p0u))
eosA = out["eos_acts"][si_mlp2].astype(np.float64)
eosu = eosA / np.linalg.norm(eosA, axis=1, keepdims=True)
Ce = eosu @ eosu.T
mean_pair_eos = (Ce.sum() - len(eosu)) / (len(eosu) ** 2 - len(eosu))
mu0 = p0u.mean(0); mu0 /= np.linalg.norm(mu0)
mue = eosu.mean(0); mue /= np.linalg.norm(mue)
print(f"\nafter MLP 2: mean pairwise cos within pos-0 vectors = {mean_pair:.3f} "
      f"(n={len(p0u)}); within mid-seq EOS vectors = {mean_pair_eos:.3f} "
      f"(n={len(eosu)}); cos(mean pos-0 dir, mean EOS dir) = {float(mu0 @ mue):.3f}")

# ---------------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
ax1, ax2, ax3 = axes
xs = np.arange(n_stages)
short = [s.replace("after ", "").replace("attention", "attn") for s in stages]

ax1.plot(xs, auc_lda[:, 0], "o-", color="tab:orange", label="LDA probe (best λ)")
ax1.plot(xs, auc_md[:, 0], "s--", color="tab:blue", label="mean-difference probe")
ax1.axhline(0.5, color="k", ls=":", lw=1, label="chance")
ax1.set_xticks(xs, short, rotation=45, ha="right")
ax1.set_ylim(0.45, 1.02)
ax1.set_ylabel("cross-validated AUC")
ax1.set_title("position 0 vs bulk: probe AUC by stage", fontsize=11)
ax1.legend(fontsize=9, frameon=False)

im = ax2.imshow(auc_lda, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
ax2.set_xticks(range(len(EARLY)), [str(p) for p in EARLY])
ax2.set_yticks(xs, short)
ax2.set_xlabel("chunk position p (vs bulk ≥ 128)")
ax2.set_title("LDA probe AUC, all stages × early positions", fontsize=11)
for si in range(n_stages):
    for pi in range(len(EARLY)):
        v = auc_lda[si, pi]
        ax2.text(pi, si, f"{v:.2f}"[1:] if v < 0.995 else "1.0", ha="center",
                 va="center", fontsize=6.5, color="w" if v < 0.85 else "k")
fig.colorbar(im, ax=ax2, shrink=0.8)

for pi, p in enumerate(EARLY):
    if p in (0, 1, 2, 4, 8, 16, 64):
        ax3.plot(xs, np.maximum(dp_lda[:, pi], 1e-2), "o-", ms=4, lw=1.2,
                 label=f"p={p}")
ax3.set_yscale("log")
ax3.set_xticks(xs, short, rotation=45, ha="right")
ax3.set_ylabel("held-out d′ (log)")
ax3.set_title("separation strength d′ by stage (AUC saturates, d′ doesn't)",
              fontsize=11)
ax3.legend(fontsize=8, frameon=False, ncol=2)
for ax in (ax1, ax3):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"pile_4l — linear readability of \"I am chunk position p\" from the "
             f"residual stream ({N_ROWS} Pile rows; bulk class = {N_BULK} random "
             f"positions ≥ {BULK_MIN} per row; EOS excluded; {N_FOLDS}-fold CV by row)",
             fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_probe.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'pos0_probe.png'}")
