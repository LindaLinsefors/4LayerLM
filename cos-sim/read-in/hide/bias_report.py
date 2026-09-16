"""Render cos-sim/read-in/bias_direction_report.md: locate the residual-stream
bias direction at each stream position (norm site) and test whether projecting
it out of the read-in components makes the within-matrix signed-cosine
distributions more symmetric / closer to the random null.

Part 1 — bias directions.  From cache/stream_stats_{pile4l,sink45}.npz
(stream_stats.py: mean mu and covariance Sigma of the post-RMSNorm input
x = g * h/rms(h) at each of the 8 sites h.<l>.rms_{1,2}, 255k Pile positions,
sink positions/EOS excluded), three candidates per site:
  v_min : lowest-variance eigenvector of Sigma (the user's primary guess)
  w     : ridge direction (Sigma + eps I)^-1 mu, eps = 1e-3 mean eigenvalue —
          maximizes the mean-to-std ratio (mu.d)/sqrt(d'Sigma d); this is
          "the" bias direction whenever one exists
  mu^   : the mean input direction itself
Diagnostic: ratio(d) = (mu.d)/sqrt(d'Sigma d) — how many standard deviations
of offset the direction carries.

Part 2 — projection test.  For each read matrix (q/k/v read rms_1, c_fc reads
rms_2), project each candidate out of the gauge-fixed alive read-ins V,
renormalize, and recompute the within-matrix signed cosine distribution
(figures overlay original vs projected vs analytic random null; table gives
tail counts / std / skew).

Usage: PYTHONUTF8=1 python cos-sim/read-in/hide/bias_report.py  (~2 min)
"""
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coci-heatmaps" / "hide"))
from load import load_tokenizer  # noqa: E402
from interactive_cross import comp_signs  # noqa: E402

CH = ROOT / "coci-heatmaps" / "hide" / "cache"
CD = ROOT / "compare-decomps" / "hide" / "cache"

SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")]
SHORT = [f"L{l}{m}" for l in range(4) for m in ("q", "k", "v", "c_fc")]
NORM_OF = {s: f"h.{s.split('.')[1]}.rms_{1 if 'attn' in s else 2}" for s in SITES}
NORM_SITES = [f"h.{l}.rms_{i}" for l in range(4) for i in (1, 2)]
NORM_SHORT = [f"L{l} rms_{i} ({'q/k/v' if i == 1 else 'c_fc'})"
              for l in range(4) for i in (1, 2)]
D = 768

DECOMPS = {
    "newA": dict(title="New A (p-8383f5e5, target t-9d2b8f02)", stats="pile4l",
                 uv=CH / "uv_newA.npz", acts=CH / "act_signs_newA.npz",
                 coci=CH / "coci_newA.npz",
                 tt=ROOT / "mean-ci-widget" / "hide" / "cache" / "top_tokens_newA.npz"),
    "C": dict(title="Sink decomposition C (p-d60af588, target t-87f91319)",
              stats="sink45",
              uv=CD / "uv_C.npz", acts=CD / "act_signs_C.npz",
              coci=CH / "coci_C.npz",
              tt=ROOT / "mean-ci-widget" / "hide" / "cache" / "top_tokens_C.npz"),
}


def random_cos_pdf(c, d=D):
    logZ = (math.lgamma(d / 2) - math.lgamma((d - 1) / 2)
            - 0.5 * math.log(math.pi))
    return np.exp(logZ + (d - 3) / 2 * np.log1p(-np.clip(c, -1, 1) ** 2))


def esc(t):
    return t.replace("|", "\\|").replace("`", "'")


def norm_site_dirs(tag):
    """Per norm site: dict cand -> unit direction, plus the diagnostics row."""
    z = np.load(HERE / "cache" / f"stream_stats_{tag}.npz")
    n = float(z["n"])
    dirs, rows = {}, {}
    for s in NORM_SITES:
        mu = z[f"{s}|S1"] / n
        Sig = z[f"{s}|S2"] / n - np.outer(mu, mu)
        lam, V = np.linalg.eigh(Sig)
        vmin = V[:, 0] * (np.sign(mu @ V[:, 0]) or 1.0)
        eps = 1e-3 * lam.mean()
        w = np.linalg.solve(Sig + eps * np.eye(D), mu)
        w /= np.linalg.norm(w)
        w *= (np.sign(mu @ w) or 1.0)
        muh = mu / np.linalg.norm(mu)

        def ratio(d):
            return float(mu @ d) / math.sqrt(float(d @ Sig @ d))

        dirs[s] = dict(vmin=vmin, w=w, mu=muh)
        rows[s] = dict(norm_mu=np.linalg.norm(mu), lam_min=lam[0],
                       lam_med=float(np.median(lam)),
                       r_vmin=ratio(vmin), r_w=ratio(w), r_mu=ratio(muh),
                       c_vw=float(vmin @ w), c_vm=float(vmin @ muh),
                       c_wm=float(w @ muh),
                       vrank_w=int((lam < w @ Sig @ w).sum()))
    return dirs, rows


def alive_ids(name, site):
    if name == "newA":
        return np.load(CH / "cross_alive.npz")[f"newA|{site}"].astype(np.int32)
    cc = np.load(CH / f"coci_{name}.npz")
    return np.flatnonzero(cc[f"{site}|mean"] > 1e-6).astype(np.int32)


def skew(v):
    m, s = v.mean(), v.std()
    return float(((v - m) ** 3).mean() / s ** 3)


def render(name, L):
    cfg = DECOMPS[name]
    FIG = HERE / "figures" / f"bias_{name}"
    FIG.mkdir(parents=True, exist_ok=True)
    relfig = f"hide/figures/bias_{name}"
    dirs, rows = norm_site_dirs(cfg["stats"])
    uv = np.load(cfg["uv"])
    acts = np.load(cfg["acts"])
    cc = np.load(cfg["coci"])
    tt = np.load(cfg["tt"])
    tok = load_tokenizer("pile_4l")

    L.append(f"\n\n---\n\n# {cfg['title']}\n")

    # ---- Part 1 table ------------------------------------------------------
    L.append("""
## Bias directions per stream position

Diagnostic $\\mathrm{ratio}(d) = (\\mu\\cdot d)/\\sqrt{d^\\top\\Sigma d}$ —
the constant offset along $d$ in units of the data's standard deviation along
$d$ (a strong bias direction has a large ratio). `var-rank(w)`: rank of the
variance along $w$ in the eigenvalue spectrum (0 = lowest-variance direction
of the site).
""")
    L.append("| site | ‖μ‖ | λ_min | λ_med | ratio(v_min) | ratio(w) | ratio(μ̂) | cos(v_min, w) | cos(v_min, μ̂) | cos(w, μ̂) | var-rank(w) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s, ns in zip(NORM_SITES, NORM_SHORT):
        r = rows[s]
        L.append(f"| {ns} | {r['norm_mu']:.1f} | {r['lam_min']:.1e} | "
                 f"{r['lam_med']:.1e} | {r['r_vmin']:.1f} | **{r['r_w']:.1f}** | "
                 f"{r['r_mu']:.1f} | {r['c_vw']:+.2f} | {r['c_vm']:+.2f} | "
                 f"{r['c_wm']:+.2f} | {r['vrank_w']} |")

    # persistence of w across sites
    W = np.stack([dirs[s]["w"] for s in NORM_SITES])
    CW = W @ W.T
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(CW, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(8), [n.split(" (")[0] for n in NORM_SHORT],
                  rotation=90, fontsize=8)
    ax.set_yticks(range(8), [n.split(" (")[0] for n in NORM_SHORT], fontsize=8)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{CW[i, j]:+.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(CW[i, j]) > 0.6 else "#333333")
    ax.set_title("cos(w, w) across stream positions", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "w_persistence.png", dpi=150)
    plt.close(fig)
    L.append(f"\n![w persistence]({relfig}/w_persistence.png)\n")

    # ---- read-ins, gauge-fixed --------------------------------------------
    Vs, idlist = {}, {}
    for si, site in enumerate(SITES):
        alive = alive_ids(name, site)
        sign = comp_signs(acts, site)[alive]
        Vall = uv[f"{site}|V"]
        V = (Vall if len(Vall) == len(alive) else Vall[alive]).astype(np.float32)
        V = V * sign[:, None]
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        Vs[si] = V
        idlist[si] = alive

    # does anything read w?  per-matrix |cos(V, w)| stats + global top
    reader_rows, wmed = [], {}
    for si, site in enumerate(SITES):
        w = dirs[NORM_OF[site]]["w"].astype(np.float32)
        proj = Vs[si] @ w
        wmed[si] = (float(np.median(np.abs(proj))), float(np.abs(proj).max()))
        for k in np.argsort(-np.abs(proj))[:3]:
            reader_rows.append((abs(proj[k]), proj[k], si, int(idlist[si][k])))
    reader_rows.sort(reverse=True)

    def top_tokens(si, i, n=4):
        site = SITES[si]
        tids, tci = tt[f"{site}|top_ids"][i], tt[f"{site}|top_ci"][i]
        tot = tt[f"{site}|total"][i]
        if tot <= 0:
            return "(no CI in sample)"
        out = []
        for t, c in zip(tids, tci):
            if len(out) >= n or (len(out) >= 3 and c / tot < 0.005):
                break
            out.append("`" + esc(repr(tok.decode([int(t)]))) + "`")
        return " ".join(out)

    allmed = np.median([wmed[si][0] for si in range(16)])
    allmax = max(wmed[si][1] for si in range(16))
    L.append(f"""
**Nothing reads $w$.** Median $|\\cos(V, w)|$ over all alive read-side
components: {allmed:.3f} (random baseline $\\sqrt{{2/\\pi d}} = 0.029$); the
single strongest alignment in any matrix is only {allmax:.3f}. The 5 most
aligned components (signed, activation gauge), for the record:

| component | cos(V, w) | mean CI | top tokens |
|---|---|---|---|""")
    for a, p, si, i in reader_rows[:5]:
        mci = cc[f"{SITES[si]}|mean"][i]
        L.append(f"| {SHORT[si]}:{i} | {p:+.2f} | {mci:.1e} | {top_tokens(si, i)} |")

    # who reads v_min at h.3.rms_1 (the massive-vector/boundary direction)?
    L.append("""
By contrast, the strongest readers of $v_\\min$ at `h.3.rms_1` (the
bulk-low-variance direction that is *not* the bias there):

| component | cos(V, v_min) | mean CI | top tokens |
|---|---|---|---|""")
    vm_rows = []
    for si, site in enumerate(SITES):
        if site not in ("h.3.attn.k_proj", "h.3.attn.v_proj"):
            continue
        vm = dirs["h.3.rms_1"]["vmin"].astype(np.float32)
        proj = Vs[si] @ vm
        for k in np.argsort(-np.abs(proj))[:5]:
            vm_rows.append((abs(proj[k]), proj[k], si, int(idlist[si][k])))
    vm_rows.sort(reverse=True)
    for a, p, si, i in vm_rows[:8]:
        mci = cc[f"{SITES[si]}|mean"][i]
        L.append(f"| {SHORT[si]}:{i} | {p:+.2f} | {mci:.1e} | {top_tokens(si, i)} |")

    # ---- Part 2: projection test ------------------------------------------
    CANDS = [("w", "w (ridge)"), ("vmin", "v_min"), ("mu", "μ̂")]
    bins = np.linspace(-1, 1, 201)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    stat_rows, hist_rows = [], []
    for si, site in enumerate(SITES):
        nd = dirs[NORM_OF[site]]
        V = Vs[si]
        n = len(V)
        iu = np.triu_indices(n, k=1)
        variants = {"orig": V}
        for ck, _ in CANDS:
            d = nd[ck].astype(np.float32)
            Vp = V - np.outer(V @ d, d)
            Vp /= np.linalg.norm(Vp, axis=1, keepdims=True)
            variants[ck] = Vp
        st, hs = {}, {}
        for vk, X in variants.items():
            vals = (X @ X.T)[iu]
            hs[vk] = np.histogram(vals, bins=bins)[0]
            st[vk] = (vals.std(), skew(vals),
                      int((vals > 0.4).sum()), int((vals < -0.4).sum()))
        stat_rows.append((si, st))
        hist_rows.append((si, n, len(iu[0]), hs))

    colors = {"orig": "#2a6fdb", "w": "#e07b39", "vmin": "#2e8b57", "mu": "#999999"}
    for logy, fname in ((True, "projected_within_matrix.png"),
                        (False, "projected_within_matrix_linear.png")):
        fig, axes = plt.subplots(4, 4, figsize=(13, 9.5), sharex=True)
        for si, n, nps, hs in hist_rows:
            ax = axes.flat[si]
            for vk in ("orig", "w", "vmin", "mu"):
                h = np.maximum(hs[vk], 0.5) if logy else hs[vk]
                if vk == "orig":
                    ax.stairs(h, bins, fill=True, color=colors[vk],
                              alpha=0.55, label="original")
                else:
                    ax.stairs(h, bins, color=colors[vk], lw=1.1,
                              ls="--" if vk == "vmin" else "-",
                              label=f"− {dict(CANDS)[vk]}")
            rnd = nps * random_cos_pdf(ctr) * (bins[1] - bins[0])
            ax.plot(ctr, np.maximum(rnd, 1e-3) if logy else rnd,
                    color="#333333", lw=1.0, ls=":")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.7)
            ax.set_title(f"{SHORT[si]}  (n = {n})", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if si == 0:
                ax.legend(fontsize=6.5, frameon=False)
        fig.suptitle(f"Within-matrix signed cos(V) before/after projecting out "
                     f"the bias direction ({name}); dotted = random null",
                     fontsize=11)
        fig.supxlabel("cos(V_a, V_b)", fontsize=10)
        fig.supylabel("pair count" + (" (log)" if logy else ""), fontsize=10)
        fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    L.append(f"""
## Projecting the bias direction out of the read-ins

Per matrix, each candidate direction (of the matrix's own input site) is
projected out of every alive gauge-fixed read-in, rows renormalized, and the
within-matrix signed cosine distribution recomputed.

![projected distributions]({relfig}/projected_within_matrix.png)

std × √d (random ≈ 1), skewness, and tail pair counts per variant
(orig → −w → −v_min → −μ̂), plus the matrix's median |cos(V, w)|:

| matrix | med \\|cos(V,w)\\| | std·√d | skew | pairs > 0.4 | pairs < −0.4 |
|---|---|---|---|---|---|""")
    for si, st in stat_rows:
        f_std = " → ".join(f"{st[k][0] * math.sqrt(D):.2f}" for k in
                           ("orig", "w", "vmin", "mu"))
        f_sk = " → ".join(f"{st[k][1]:+.2f}" for k in ("orig", "w", "vmin", "mu"))
        f_p = " → ".join(str(st[k][2]) for k in ("orig", "w", "vmin", "mu"))
        f_n = " → ".join(str(st[k][3]) for k in ("orig", "w", "vmin", "mu"))
        L.append(f"| {SHORT[si]} | {wmed[si][0]:.3f} | {f_std} | {f_sk} | {f_p} | {f_n} |")
    return rows, stat_rows


L = ["""# The residual-stream bias direction, located per stream position — and its role in read-in cosine similarity

**Question (user, 2026-09-16):** find the bias direction at each position in
the residual stream (primary guess: the lowest-variance direction), then
project it out of the read-in components and check whether the within-matrix
cosine distributions become more symmetric and/or closer to the random null.

**Method.** For each norm site `h.<l>.rms_1` (feeds q/k/v) and `h.<l>.rms_2`
(feeds c_fc), the statistics of the *actual read-in input*
$x = g \\odot h/\\mathrm{rms}(h)$ (gain included) over 255k Pile positions
(500 cached rows; positions 0–1 and `<|endoftext|>` excluded as
massive-vector outliers; `hide/stream_stats.py` →
`hide/cache/stream_stats_{pile4l,sink45}.npz`). The C target is run with the
**corrected RoPE** (`load.load_sink(45)`), so its stream statistics are not
affected by the 2026-09-15 loader bug (the component gauge and mean-CI
annotations still are). Candidates per site:

- $v_\\min$ — lowest-variance eigenvector of the input covariance $\\Sigma$;
- $w \\propto (\\Sigma + \\epsilon I)^{-1}\\mu$ ($\\epsilon = 10^{-3}\\bar\\lambda$)
  — maximizes $\\mathrm{ratio}(d) = (\\mu\\cdot d)/\\sqrt{d^\\top\\Sigma d}$,
  i.e. the direction whose constant offset is largest relative to how much
  the data varies along it: the natural definition of "the bias direction";
- $\\hat\\mu$ — the mean input direction.

## Headline findings

- **A strong bias direction exists at every stream position, in both
  models**: ratio(w) = 17–50 everywhere — a constant offset of tens of
  standard deviations. It always lives in the low-variance end of the
  spectrum (variance rank ≳ 745 of 768 from the top).
- **The lowest-variance direction is the bias direction only at some
  sites.** cos(v_min, w) ≈ 0.99 at L0/L2 rms_1 (both models) and most sink45
  sites — but at pile `h.1.rms_1` (the stream after block 0) v_min carries
  *no* mean at all (ratio 0.2, cos(v_min, w) = 0.01): the lowest-variance
  direction there is a frozen direction that is not the bias. This matches
  the token-embeds finding that MLP 1 destroys the embedding bias direction
  while new frozen directions appear later.
- **Projecting the bias direction out changes the within-matrix cosine
  distributions essentially not at all** (mean skew: newA +1.86 → +1.82,
  C +0.73 → +0.68; tail counts move by ~1%), **because the read-in
  components do not read it**: median |cos(V, w)| over all alive read-side
  components ≈ 0.032 in both decompositions — the random-direction baseline
  is 0.029 — and the single strongest alignment across all 16 matrices is
  only ~0.05. The positive skew and heavy tails of the cosine distributions
  are shared *feature/token* directions, not a shared bias component. (That
  a 17–50 σ constant offset is read by *no* component is itself notable —
  the decompositions route the bias around the component dictionary.)
- **The one direction whose removal does reshape a distribution is not the
  bias but the massive-vector/boundary direction**: in newA, v_min at
  `h.3.rms_1` (a *bulk*-low-variance direction — sink positions were
  excluded from the statistics — orthogonal to w there, cos 0.26) is read at
  |cos| 0.5–0.9 by exactly the known L3 EOS/pos-0 sink components; projecting
  it out collapses the L3k tail (pairs > 0.4: 83 → 18, skew +3.7 → +1.5) and
  L3v (14 → 2). In C the effect is absent — consistent with the built-in
  sinks removing that machinery.
"""]

summary = {}
for nm in ("newA", "C"):
    summary[nm] = render(nm, L)

L.append("""
---

## Linear-scale versions

The same projection distribution plots with a linear y axis (the log plots
emphasize the tails; these show where the actual mass sits).

**New A:**

![newA projected linear](hide/figures/bias_newA/projected_within_matrix_linear.png)

**Sink decomposition C:**

![C projected linear](hide/figures/bias_C/projected_within_matrix_linear.png)
""")

(ROOT / "cos-sim" / "read-in" / "bias_direction_report.md").write_text(
    "\n".join(L), encoding="utf-8")
print("report written -> cos-sim/read-in/bias_direction_report.md")
for nm, (rows, stat_rows) in summary.items():
    o = [st["orig"] for _, st in stat_rows]
    w = [st["w"] for _, st in stat_rows]
    print(f"{nm}: mean skew orig {np.mean([x[1] for x in o]):+.2f} -> -w "
          f"{np.mean([x[1] for x in w]):+.2f}; pairs>0.4 "
          f"{sum(x[2] for x in o)} -> {sum(x[2] for x in w)}; pairs<-0.4 "
          f"{sum(x[3] for x in o)} -> {sum(x[3] for x in w)}")
