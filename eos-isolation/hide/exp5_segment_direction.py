"""Experiment 5 — the segment direction: what in the residual stream separates
pre-EOS from post-EOS tokens?

Rows with exactly one <|endoftext|> (at position p). Classes: pre = positions
2..p-1, post = p+1..511 (the EOS itself and sink positions 0/1 excluded). At
each of 10 stream stages (after embedding, after each attention, after each
MLP, after ln_f) fit the Fisher/LDA direction on a random half of the rows,

    w = Sigma_w^{-1} (mu_post - mu_pre),   Sigma_w = pooled within-class cov
                                           (+ 1e-3 tr/d ridge), unit-normalized,
    sign convention: post-EOS mean projection >= 0,

then histogram the *held-out* rows' projections per class and report AUC / d'.

Two floors calibrate the numbers: the after-embedding stage separates only via
token statistics (doc openers are enriched in 'Q', 'The', ... — no context in
the embeddings), and an identical pipeline on EOS-free rows with matched
pseudo-boundaries measures how much a pure *position* code can separate the
classes (pre/post correlates with absolute position by construction).

Cache: hide/cache/exp5.npz. Figure: ../exp5_segment_direction.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import C_DOC, C_PRE, EOS_ID, HERE, ROOT, T, load_rows, style

import sys
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 8
LAM_REL = 1e-3
N_KEEP = 40_000        # held-out projections kept per class/stage for the histograms
CACHE = HERE / "cache" / "exp5.npz"
STAGES = (["after embedding"]
          + [s for l in range(4) for s in (f"after attn {l + 1}", f"after MLP {l + 1}")]
          + ["after ln_f"])
NS = len(STAGES)


def stage_tensors(model, caps):
    """The 10 residual-stream stages from the captured hook tensors."""
    out = [caps["in0"]]
    for l in range(4):
        out.append(caps[f"in{l}"] + caps[f"attn{l}"])
        out.append(caps[f"out{l}"])
    out.append(caps["final"])
    return out


def run_cohort(model, rows_t, ps, fit_mask):
    """Accumulate LDA sufficient statistics on fit rows and return, per stage,
    (mu0, mu1, Sw) plus a function to stream held-out projections later.
    rows_t: (N, T) tokens; ps: boundary position per row; fit_mask: bool (N,)."""
    caps = {}
    hooks = [model.h[l].register_forward_pre_hook(
        lambda _m, inp, l=l: caps.__setitem__(f"in{l}", inp[0])) for l in range(4)]
    hooks += [model.h[l].attn.register_forward_hook(
        lambda _m, _i, o, l=l: caps.__setitem__(f"attn{l}", o)) for l in range(4)]
    hooks += [model.h[l].register_forward_hook(
        lambda _m, _i, o, l=l: caps.__setitem__(f"out{l}", o)) for l in range(4)]
    hooks.append(model.ln_f.register_forward_hook(
        lambda _m, _i, o: caps.__setitem__("final", o)))

    pos = torch.arange(T, device=DEVICE)
    d = model.config.n_embd
    n = np.zeros((NS, 2))
    s = np.zeros((NS, 2, d))
    G = np.zeros((NS, 2, d, d))

    def masks(p):
        pre = (pos >= 2) & (pos < p)
        post = pos > p
        return pre, post

    with torch.no_grad():
        for b0 in range(0, int(fit_mask.sum()), BS):
            idx = np.nonzero(fit_mask)[0][b0:b0 + BS]
            batch = rows_t[idx].to(DEVICE)
            model(batch)
            stages = stage_tensors(model, caps)
            for si, x in enumerate(stages):
                for b, ri in enumerate(idx):
                    for ci, m in enumerate(masks(int(ps[ri]))):
                        sel = x[b, m].float()
                        n[si, ci] += len(sel)
                        s[si, ci] += sel.sum(0).double().cpu().numpy()
                        G[si, ci] += (sel.T @ sel).double().cpu().numpy()
    for h in hooks:
        h.remove()

    ws = np.zeros((NS, d))
    for si in range(NS):
        mu = s[si] / n[si][:, None]
        Sw = sum(G[si, ci] - n[si, ci] * np.outer(mu[ci], mu[ci]) for ci in (0, 1))
        Sw /= n[si].sum()
        lam = LAM_REL * np.trace(Sw) / d
        w = np.linalg.solve(Sw + lam * np.eye(d), mu[1] - mu[0])
        w /= np.linalg.norm(w)
        if (mu[1] - mu[0]) @ w < 0:
            w = -w
        ws[si] = w
    return ws


def project_cohort(model, rows_t, ps, test_idx, ws):
    caps = {}
    hooks = [model.h[l].register_forward_pre_hook(
        lambda _m, inp, l=l: caps.__setitem__(f"in{l}", inp[0])) for l in range(4)]
    hooks += [model.h[l].attn.register_forward_hook(
        lambda _m, _i, o, l=l: caps.__setitem__(f"attn{l}", o)) for l in range(4)]
    hooks += [model.h[l].register_forward_hook(
        lambda _m, _i, o, l=l: caps.__setitem__(f"out{l}", o)) for l in range(4)]
    hooks.append(model.ln_f.register_forward_hook(
        lambda _m, _i, o: caps.__setitem__("final", o)))
    pos = torch.arange(T, device=DEVICE)
    ws_t = torch.tensor(ws, dtype=torch.float32, device=DEVICE)
    proj = [[[], []] for _ in range(NS)]
    with torch.no_grad():
        for b0 in range(0, len(test_idx), BS):
            idx = test_idx[b0:b0 + BS]
            batch = rows_t[idx].to(DEVICE)
            model(batch)
            stages = stage_tensors(model, caps)
            for si, x in enumerate(stages):
                pr = x.float() @ ws_t[si]                     # (B, T)
                for b, ri in enumerate(idx):
                    p = int(ps[ri])
                    proj[si][0].append(pr[b, (pos >= 2) & (pos < p)].cpu().numpy())
                    proj[si][1].append(pr[b, pos > p].cpu().numpy())
    for h in hooks:
        h.remove()
    return [[np.concatenate(c) if c else np.zeros(0) for c in st] for st in proj]


def auc_d(x0, x1):
    """rank AUC of x1 > x0, and d'."""
    lab = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    r = np.argsort(np.argsort(np.r_[x0, x1])) + 1.0
    auc = (r[lab == 1].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x0) * len(x1))
    dp = (x1.mean() - x0.mean()) / np.sqrt(0.5 * (x0.var() + x1.var()))
    return auc, dp


def compute():
    rows = load_rows()
    e = (rows == EOS_ID).numpy()
    single = np.nonzero(e.sum(1) == 1)[0]
    ps_all = e[single].argmax(1)
    ok = ps_all >= 2                       # need at least some pre tokens? keep all with p>=2
    single, ps_s = single[ok], ps_all[ok]
    print(f"{len(single)} single-EOS rows, p range {ps_s.min()}..{ps_s.max()}")

    rng = np.random.default_rng(0)
    eos_free = np.nonzero(~e.any(1))[0]
    ctrl = rng.choice(eos_free, len(single), replace=False)
    ps_c = rng.choice(ps_s, len(single))   # matched pseudo-boundary distribution

    model, _, _ = load_pile_4l()
    model = model.to(DEVICE)

    out = {}
    for name, ridx, ps in (("real", single, ps_s), ("ctrl", ctrl, ps_c)):
        rows_t = rows[ridx]
        fit = np.zeros(len(ridx), bool)
        fit[rng.permutation(len(ridx))[:len(ridx) // 2]] = True
        # fit directions on the fit half (statistics pass)
        ws = run_cohort(model, rows_t, ps, fit)
        proj = project_cohort(model, rows_t, ps, np.nonzero(~fit)[0], ws)
        aucs, dps = np.zeros(NS), np.zeros(NS)
        keep = np.zeros((NS, 2, N_KEEP), np.float32)
        kn = np.zeros((NS, 2), int)
        for si in range(NS):
            aucs[si], dps[si] = auc_d(proj[si][0], proj[si][1])
            for ci in (0, 1):
                x = proj[si][ci]
                if len(x) > N_KEEP:
                    x = rng.choice(x, N_KEEP, replace=False)
                keep[si, ci, :len(x)] = x
                kn[si, ci] = len(x)
        out[f"{name}_w"] = ws
        out[f"{name}_auc"] = aucs
        out[f"{name}_dp"] = dps
        out[f"{name}_proj"] = keep
        out[f"{name}_projn"] = kn
        print(f"{name}: AUC per stage " + " ".join(f"{a:.3f}" for a in aucs))
    out["ps"] = ps_s
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"saved {CACHE}")


if not CACHE.exists():
    compute()
z = np.load(CACHE)

print("\nstage                AUC     d'     ctrl AUC   cos(w, w_prev)")
for si, st in enumerate(STAGES):
    cosp = abs(float(z["real_w"][si] @ z["real_w"][si - 1])) if si else np.nan
    print(f"{st:18s}  {z['real_auc'][si]:.3f}  {z['real_dp'][si]:6.2f}   {z['ctrl_auc'][si]:.3f}"
          f"      {cosp:.2f}" if si else
          f"{st:18s}  {z['real_auc'][si]:.3f}  {z['real_dp'][si]:6.2f}   {z['ctrl_auc'][si]:.3f}")

# cos with the massive-vector direction u (defined after MLP 2)
u = np.load(ROOT / "endoftext-pos0/hide/cache/pos0_components.npz")["u"]
short = ["emb"] + [f"{k}{l + 1}" for l in range(4) for k in ("att", "mlp")] + ["lnf"]
print("\n|cos(w_stage, massive direction u)|: "
      + "  ".join(f"{sh}:{abs(z['real_w'][si] @ u):.2f}"
                  for si, sh in enumerate(short)))

# ---- figure ----
fig, axes = plt.subplots(2, 5, figsize=(17, 6.8))
for si, (st, ax) in enumerate(zip(STAGES, axes.flat)):
    x0 = z["real_proj"][si, 0, :z["real_projn"][si, 0]]
    x1 = z["real_proj"][si, 1, :z["real_projn"][si, 1]]
    lo, hi = np.percentile(np.r_[x0, x1], [0.2, 99.8])
    bins = np.linspace(lo, hi, 120)
    ax.hist(x0, bins=bins, density=True, color=C_PRE, alpha=0.6,
            label="pre-EOS tokens")
    ax.hist(x1, bins=bins, density=True, color=C_DOC, alpha=0.6,
            label="post-EOS tokens")
    ax.set_title(f"{st}\nAUC {z['real_auc'][si]:.3f}  (position ctrl "
                 f"{z['ctrl_auc'][si]:.3f})", fontsize=9.5)
    ax.set_yticks([])
    style(ax)
    if si == 0:
        ax.legend(fontsize=8, frameon=False)
    if si >= 5:
        ax.set_xlabel("projection onto w (unit norm)", fontsize=9)
fig.suptitle("pile_4l — the segment direction: held-out projections onto the LDA "
             "direction separating pre- from post-EOS tokens\n"
             f"({len(z['ps'])} single-EOS rows; EOS + positions 0-1 excluded; "
             "position ctrl = same pipeline on EOS-free rows with matched pseudo-boundaries)",
             y=1.00, fontsize=11)
fig.tight_layout()
fig.savefig(HERE.parent / "exp5_segment_direction.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'exp5_segment_direction.png'}")
