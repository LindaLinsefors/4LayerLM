"""Test the variance/bimodality (parity-code) hypothesis for the position-0
signal: are there directions w in the pre-o_proj value-mixture space where
per-token values are +-c-like, so that

  pos 0:  w.y = w.v(t_0)          -> full spread, ideally bimodal +-c
  bulk:   w.y = sum_j a_j w.v_j   -> concentrated near 0 by averaging

i.e. a high-variance-ratio signal with no mean shift, readable only through an
even nonlinearity (an MLP), not by a linear probe?

Method, per layer 0 and 1:
  1. capture y = o_proj input (the concatenated per-head value mixtures,
     768-d) at positions {0,1,2,4,8,16,32,64} + 4 random bulk >= 128 per row
  2. generalized eigenproblem  Sigma_pos0 w = lambda (Sigma_bulk + eps I) w:
     directions sorted by variance ratio lambda = var(w.y|pos0)/var(w.y|bulk).
     Any direction has lambda > 1 from generic averaging (CLT shrinkage);
     a dedicated code shows up as an outlier tail + bimodality.
  3. for the top directions: histogram of w.y at pos 0/1/4/bulk, variance vs
     position, Pearson kurtosis at pos 0 (two-point +-c -> 1, Gaussian -> 3),
     per-head energy of w, AUC of the folded statistic |w.y| (readable by an
     even nonlinearity) vs the linear AUC of w.y.
  4. component check: for every fired o_proj component (read-in V_c applied to
     y), variance ratio + kurtosis + mean-shift z of s_c = V_c.y at pos 0 vs
     bulk; CI-early-locked components (from pos0_ci.npz) highlighted, plus
     cos^2 overlap of V_c with the span of the top-10 variance directions.

Data: first N_ROWS cached Pile rows; EOS positions excluded everywhere.
Output: endoftext-pos0/pos0_variance_signal.png + printed tables.
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
EARLY = [0, 1, 2, 4, 8, 16, 32, 64]
N_BULK = 4
BULK_MIN = 128
EPS_FRAC = 1e-4          # ridge on Sigma_bulk, as fraction of mean diag
N_HEAD, HD = 6, 128

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]
rng = np.random.default_rng(0)
bulk_pos = rng.integers(BULK_MIN, T, size=(N_ROWS, N_BULK))

model, pc, _ = load_pile_4l()
model = model.to(DEVICE)

caps: dict[int, torch.Tensor] = {}
hooks = [model.h[l].attn.o_proj.register_forward_pre_hook(
    lambda _m, inp, l=l: caps.__setitem__(l, inp[0])) for l in (0, 1)]

n_keep = len(EARLY) + N_BULK
Y = {l: np.zeros((N_ROWS, n_keep, 768), dtype=np.float32) for l in (0, 1)}
kept_tok = np.zeros((N_ROWS, n_keep), dtype=np.int64)
with torch.no_grad():
    for start in range(0, N_ROWS, BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        tok = batch.cpu().numpy()
        for bi in range(len(tok)):
            ri = start + bi
            pos = np.concatenate([np.array(EARLY), bulk_pos[ri]])
            kept_tok[ri] = tok[bi, pos]
            for l in (0, 1):
                Y[l][ri] = caps[l][bi, pos].float().cpu().numpy()
for h in hooks:
    h.remove()

bulk_slots = slice(len(EARLY), None)
bulk_ok = kept_tok[:, bulk_slots] != EOS_ID


def kurt(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def auc(s0, s1):
    r = np.argsort(np.argsort(np.concatenate([s0, s1]))).astype(float) + 1
    return (r[len(s0):].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s0) * len(s1))


results = {}
for l in (0, 1):
    A = Y[l].astype(np.float64)
    ok0 = kept_tok[:, 0] != EOS_ID
    X0 = A[ok0, 0]                                # pos-0 mixtures = raw self values
    Xb = A[:, bulk_slots][bulk_ok]
    mu0, mub = X0.mean(0), Xb.mean(0)
    S0 = np.cov(X0.T)
    Sb = np.cov(Xb.T)
    eps = EPS_FRAC * np.trace(Sb) / 768
    # generalized eigenproblem via whitening: B^{-1/2} S0 B^{-1/2}
    evalb, evecb = np.linalg.eigh(Sb + eps * np.eye(768))
    Bm12 = evecb @ np.diag(evalb ** -0.5) @ evecb.T
    lam, W = np.linalg.eigh(Bm12 @ S0 @ Bm12)
    lam, W = lam[::-1], (Bm12 @ W)[:, ::-1]       # descending; columns = directions
    W /= np.linalg.norm(W, axis=0)

    proj = {"pos0": X0 @ W[:, :10]}
    for pi, p in enumerate(EARLY[1:], start=1):
        okp = kept_tok[:, pi] != EOS_ID
        proj[p] = A[okp, pi] @ W[:, :10]
    proj["bulk"] = Xb @ W[:, :10]

    k0 = [kurt(proj["pos0"][:, i]) for i in range(10)]
    head_energy = (W[:, :3].reshape(N_HEAD, HD, 3) ** 2).sum(1)   # (6, 3)
    lin_auc = [auc(proj["bulk"][:, i], proj["pos0"][:, i]) for i in range(3)]
    fold_auc = [auc(np.abs(proj["bulk"][:, i] - proj["bulk"][:, i].mean()),
                    np.abs(proj["pos0"][:, i] - proj["bulk"][:, i].mean()))
                for i in range(3)]

    print(f"\n=== layer {l} (o_proj input y) ===")
    print(f"variance-ratio spectrum: max {lam[0]:.0f}, top-10 "
          f"{np.array2string(lam[:10], precision=0, floatmode='fixed')}, "
          f"median {np.median(lam):.1f}, min {lam[-1]:.2f}")
    print(f"top-10 pos-0 kurtosis (2-point=1, Gauss=3): "
          f"{np.array2string(np.array(k0), precision=2)}")
    print("top-3 directions: linear AUC (w.y) vs folded AUC (|w.y - mean_bulk|):")
    for i in range(3):
        print(f"  dir {i}: ratio {lam[i]:7.0f}  lin {lin_auc[i]:.3f}  "
              f"fold {fold_auc[i]:.3f}  head energy "
              + " ".join(f"h{h}:{head_energy[h, i]:.2f}" for h in range(N_HEAD)))
    results[l] = dict(lam=lam, W=W, proj=proj, k0=k0, mu0=mu0, mub=mub,
                      S0=S0, Sb=Sb, X0=X0, Xb=Xb)

# ---------------------------------------------------- component check
ci = np.load(HERE / "cache" / "pos0_ci.npz")
comp_stats = {}
for l in (0, 1):
    m = f"h.{l}.attn.o_proj"
    Vc = pc.components[m].V.double().numpy()      # (768, C)
    Fp = ci[f"{m}|Fp"]
    F = Fp.sum(1)
    with np.errstate(invalid="ignore"):
        efrac = Fp[:, :8].sum(1) / F
    fired = np.where(F > 0)[0]
    early = np.where((F >= 50) & (efrac > 0.5))[0]

    r = results[l]
    s0 = r["X0"] @ Vc                             # (n0, C)
    sb = r["Xb"] @ Vc
    var_ratio = s0.var(0) / sb.var(0)
    zshift = np.abs(s0.mean(0) - sb.mean(0)) / sb.std(0)
    kurt0 = np.array([kurt(s0[:, c]) for c in range(Vc.shape[1])])
    Wtop = r["W"][:, :10]
    Vn = Vc / np.linalg.norm(Vc, axis=0)
    cos2 = ((Vn.T @ np.linalg.qr(Wtop)[0]) ** 2).sum(1)
    comp_stats[l] = dict(fired=fired, early=early, var_ratio=var_ratio,
                         zshift=zshift, kurt0=kurt0, cos2=cos2, F=F)

    print(f"\n{m}: fired {len(fired)}, CI-early-locked {len(early)}")
    print(f"  median var-ratio: early-locked {np.median(var_ratio[early]):.1f} "
          f"vs other fired {np.median(var_ratio[np.setdiff1d(fired, early)]):.1f}")
    print(f"  median |mean-shift| z: early {np.median(zshift[early]):.2f} "
          f"vs other {np.median(zshift[np.setdiff1d(fired, early)]):.2f}")
    print(f"  median pos-0 kurtosis: early {np.median(kurt0[early]):.2f} "
          f"vs other {np.median(kurt0[np.setdiff1d(fired, early)]):.2f}")
    print(f"  median cos^2 with top-10 var-dir span: early "
          f"{np.median(cos2[early]):.3f} vs other "
          f"{np.median(cos2[np.setdiff1d(fired, early)]):.3f} (chance 10/768 = 0.013)")
    top = early[np.argsort(-var_ratio[early])][:8]
    print("  top early-locked by var-ratio:  comp | var-ratio | z-shift | kurt0 | cos2 | fires")
    for c in top:
        print(f"    {c:5d} {var_ratio[c]:9.1f} {zshift[c]:8.2f} {kurt0[c]:6.2f} "
              f"{cos2[c]:5.2f} {int(F[c]):8d}")

# ---------------------------------------------------------------------- plot
fig, axes = plt.subplots(2, 4, figsize=(17.5, 9))
for l in (0, 1):
    r = results[l]
    ax_sp, ax_hist, ax_var, ax_comp = axes[l]

    ax_sp.semilogy(r["lam"], lw=1.2)
    ax_sp.axhline(np.median(r["lam"]), color="k", ls=":", lw=0.8,
                  label=f"median {np.median(r['lam']):.1f} (generic averaging)")
    ax_sp.set_xlabel("direction rank")
    ax_sp.set_ylabel("var(w·y | pos 0) / var(w·y | bulk)")
    ax_sp.set_title(f"L{l}: variance-ratio spectrum of the value mixture", fontsize=10)
    ax_sp.legend(fontsize=8, frameon=False)

    w0 = 0
    bins = np.linspace(*np.percentile(np.concatenate(
        [r["proj"]["pos0"][:, w0], r["proj"]["bulk"][:, w0]]), [0.2, 99.8]), 60)
    for key, col in [("bulk", "tab:blue"), (4, "tab:green"),
                     (1, "tab:purple"), ("pos0", "tab:orange")]:
        ax_hist.hist(r["proj"][key][:, w0], bins=bins, density=True, alpha=0.55,
                     color=col, label=f"pos {key}" if key != "bulk" else "bulk")
    ax_hist.set_xlabel("w·y (top variance-ratio direction)")
    ax_hist.set_title(f"L{l}: top direction, kurtosis at pos 0 = "
                      f"{r['k0'][0]:.2f} (±c code → 1, Gauss → 3)", fontsize=10)
    ax_hist.legend(fontsize=8, frameon=False)

    ps = EARLY
    for i in range(3):
        vs = [r["proj"]["pos0"][:, i].var()] + \
             [r["proj"][p][:, i].var() for p in EARLY[1:]]
        ax_var.plot(ps, vs / r["proj"]["bulk"][:, i].var(), "o-", ms=4,
                    label=f"dir {i} (ratio {r['lam'][i]:.0f})")
    ax_var.axhline(1, color="k", ls=":", lw=0.8)
    ax_var.set_yscale("log")
    ax_var.set_xlabel("chunk position p")
    ax_var.set_ylabel("var at p / var at bulk")
    ax_var.set_title(f"L{l}: variance profile of top directions", fontsize=10)
    ax_var.legend(fontsize=8, frameon=False)

    cs = comp_stats[l]
    other = np.setdiff1d(cs["fired"], cs["early"])
    ax_comp.loglog(np.maximum(cs["zshift"][other], 1e-3), cs["var_ratio"][other],
                   ".", ms=4, alpha=0.5, color="tab:gray", label="fired o comps")
    ax_comp.loglog(np.maximum(cs["zshift"][cs["early"]], 1e-3),
                   cs["var_ratio"][cs["early"]], ".", ms=6, color="tab:red",
                   label="CI early-locked")
    ax_comp.set_xlabel("|mean shift| / bulk std of s_c (linear signal)")
    ax_comp.set_ylabel("var ratio pos0/bulk of s_c")
    ax_comp.set_title(f"L{l} o_proj components: variance code vs mean code", fontsize=10)
    ax_comp.legend(fontsize=8, frameon=False)

for ax in axes.flat:
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — testing the variance/bimodality (±code) hypothesis for "
             f"the position-0 signal in the pre-o_proj value mixture "
             f"({N_ROWS} Pile rows)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_variance_signal.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {HERE.parent / 'pos0_variance_signal.png'}")

np.savez(HERE / "cache" / "pos0_variance_signal.npz",
         **{f"L{l}_{k}": results[l][k] for l in (0, 1) for k in ("lam", "W")},
         **{f"L{l}_comp_{k}": comp_stats[l][k] for l in (0, 1)
            for k in ("var_ratio", "zshift", "kurt0", "cos2", "early")})
print(f"saved {HERE / 'cache' / 'pos0_variance_signal.npz'}")
