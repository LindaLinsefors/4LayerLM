"""Render the SIGNED write-out-cosine reports: cos-sim/write-out/newA_report.md
and cos-sim/write-out/C_report.md (+ figures) from cache/ucos_signed_{decomp}.npz.

Write-side mirror of cos-sim/read-in/hide/report_signed.py: clusters the
cos(U) > 0.7 graph over the pooled write-side alive components (o_proj +
down_proj), negative tail at cos < -0.4, members annotated with mean CI,
pos-0 fire share, unembedding alignment and top activating tokens.
Run with PYTHONUTF8=1.

Usage: python cos-sim/write-out/hide/report_signed.py [newA C]  (default: both)
Inputs per decomposition: cache/ucos_signed_*.npz (compute_signed.py),
cache/cluster_ci_signed_*.npz (cluster_ci_signed_modal.py), the uv/act_signs/
coci/top_tokens/pos-fires caches listed in DECOMPS, and the target-model
safetensors (unembedding for logit alignment).
"""
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coci-heatmaps" / "hide"))
from load import load_tokenizer  # noqa: E402
from interactive_cross import comp_signs  # noqa: E402

CH = ROOT / "coci-heatmaps" / "hide" / "cache"
CD = ROOT / "compare-decomps" / "hide" / "cache"

SITES = [f"h.{l}.{m}" for l in range(4) for m in ("attn.o_proj", "mlp.down_proj")]
SHORT = [f"L{l}{m}" for l in range(4) for m in ("o", "down")]
NSITES = len(SITES)
THR = 0.7
NEG_THR = -0.4
D = 768
BASELINE = np.sqrt(2 / (np.pi * D))   # E|cos| for random unit vectors

DECOMPS = {
    "newA": dict(
        title="New A", run="p-8383f5e5", target="t-9d2b8f02",
        report="newA_report.md",
        uv=CH / "uv_newA.npz", acts=CH / "act_signs_newA.npz",
        coci=CH / "coci_newA.npz",
        tt=ROOT / "mean-ci-widget" / "hide" / "cache" / "top_tokens_newA.npz",
        pf=CH / "pos_fires_newA.npz",
        st=ROOT / "prev_paper" / "models" / "pile_4layer"
        / "target_model_t-9d2b8f02" / "model_step_99999.safetensors",
        unemb="wte.weight", tied=True),
    "C": dict(
        title="Sink decomposition C", run="p-d60af588", target="t-87f91319",
        report="C_report.md",
        uv=CD / "uv_C.npz", acts=CD / "act_signs_C.npz",
        coci=CH / "coci_C.npz",
        tt=ROOT / "mean-ci-widget" / "hide" / "cache" / "top_tokens_C.npz",
        pf=ROOT / "sink-models" / "hide" / "cache" / "pos_fires_sink_C.npz",
        st=ROOT / "sink-models" / "pretrain_cache" / "spd-t-87f91319"
        / "model_step_100000.safetensors",
        unemb="lm_head.weight", tied=False),
}

# hand-written decomposition-specific commentary, inserted into Headlines
NOTES = {
    "newA": ["""
- **Write-direction sharing is ~4× rarer than read-direction sharing.** 263
  pairs above +0.7 (2% of components clustered) vs the read side's 1,089
  (11% clustered) on comparable pool sizes — components far more often read
  a common feature than write into a common channel.
- **Unlike the read side, the negative tail is a real population** — 1,790
  pairs < −0.4 and 148 < −0.7 (read side: 217 and 8) — and its strongest
  pairs are the boundary/sink machinery caught in the act of *write-then-
  cancel*: the L1 down_proj structural-boundary writers (2889, 778, 2028,
  486, 613 — top tokens `'\\n'` `'.'` `' the'`) sit at cos −0.86..−0.95
  against L3 o_proj/down_proj EOS/pos-0 components that **co-fire with them**
  (co-CI r 0.79–0.97): anti-aligned writes on the same tokens, i.e. the
  layer-3 side actively cancels what layer 1 wrote — matching the
  endoftext-pos0 finding that attention 4 / late layers cancel the massive
  boundary vector before the unembedding.
- **Cluster 1 (26 comps, L2o/L2down/L3o/L3down, pos-0/EOS-dominated) is the
  cancellation side bundled**: the many components that co-write one shared
  direction opposite the L1 boundary writers.
- **No unembedding alignment anywhere**: layer medians +0.03/+0.00/+0.00/
  −0.02 — writers do not write along their own trigger token's unembedding,
  even in L3 (one step before the logits). Expected: the top activating token
  is the *input* trigger, and promoting it would predict repetition.
- **Aligned writers co-fire strongly but read near-orthogonally** (same-matrix
  median co-CI r 0.83, median cos(V) +0.03) — the mirror image of the read
  side (aligned readers write orthogonally, median |cos U| 0.08): shared
  write channel, split input features. The one true near-duplicate:
  L0down:1740 ↔ L0down:2092 (`http`/`https`, r 1.00, cos V +0.83, cos U
  +0.78).
"""],
    "C": ["""
- **Write-direction sharing is essentially absent in C**: 7 pairs above +0.7
  out of 38.6M (newA: 263), 2 clusters totalling 7 components, max cos 0.90.
  The read side already showed C sharing an order of magnitude less than
  newA; on the write side the collapse is even more complete.
- **The negative tail is boundary machinery again, but WITHOUT co-firing.**
  Nearly every strong negative pair joins `'\\n'`/`'.'`-firing components
  across L0–L3 o/down matrices — yet their co-CI r ≈ 0.00 throughout (newA's
  strongest negative pairs co-fired at r 0.8–0.97). There is no co-firing
  write-then-cancel geometry — consistent with the built-in attention sinks
  removing the emergent massive-vector write+cancel circuit these
  decompositions otherwise learn.
- **The two same-matrix aligned pairs behave read-side-ish**: co-CI r ≈ 0.02
  with cos V +0.44 — nothing like newA's tightly co-firing write clusters.
- **⚠ RoPE caveat (2026-09-15):** the cos(U)/cos(V) geometry is pure weights
  and unaffected, but everything CI-derived for C — alive sets, mean CI, top
  tokens, pos-0 shares, the activation-sign gauge, and all co-CI r here
  (`cluster_ci_signed_C.npz` was computed through the public sink loader) —
  used the broken-RoPE forward (`sink-models/rope_report.md`).
  Token-identity/short-range statistics likely survive qualitatively;
  recompute pending the correct RoPE convention.
"""],
}


def esc(t):
    # tokens go inside code spans: literal except table pipes (GFM special case)
    return t.replace("|", "\\|").replace("`", "'")


def random_cos_pdf(c, d=D):
    """Density of cos(angle) between two random directions in R^d."""
    logZ = (math.lgamma(d / 2) - math.lgamma((d - 1) / 2)
            - 0.5 * math.log(math.pi))
    return np.exp(logZ + (d - 3) / 2 * np.log1p(-np.clip(c, -1, 1) ** 2))


def render(name):
    cfg = DECOMPS[name]
    FIG = HERE / "figures" / f"signed_{name}"
    FIG.mkdir(parents=True, exist_ok=True)
    CFIG = FIG / "clusters"
    CFIG.mkdir(exist_ok=True)
    relfig = f"hide/figures/signed_{name}"

    z = np.load(HERE / "cache" / f"ucos_signed_{name}.npz")
    site, ids = z["site"], z["ids"]
    N = len(site)
    P = z["pairs"]
    n_pairs_tot = N * (N - 1) // 2
    tok = load_tokenizer("pile_4l")
    tt = np.load(cfg["tt"])
    cc = np.load(cfg["coci"])
    pf = np.load(cfg["pf"])
    uv = np.load(cfg["uv"])
    acts = np.load(cfg["acts"])
    signs = {s: comp_signs(acts, SITES[s]) for s in range(NSITES)}

    # row lookup into the uv dump: newA holds only alive rows (in the pooled
    # id order), C holds all C rows in id order
    row_of = {}
    for s in range(NSITES):
        a = ids[site == s]
        full = len(uv[f"{SITES[s]}|U"]) == cc[f"{SITES[s]}|mean"].shape[0]
        row_of[s] = ({int(c): int(c) for c in a} if full
                     else {int(c): k for k, c in enumerate(a)})

    st_ = safe_open(cfg["st"], framework="np")
    unemb = st_.get_tensor(cfg["unemb"]).astype(np.float32)
    unemb /= np.linalg.norm(unemb, axis=1, keepdims=True)
    # gauge-fixed unit write directions, pooled order (matches compute_signed.py)
    Un = np.concatenate(
        [uv[f"{SITES[s]}|U"][[row_of[s][int(c)] for c in ids[site == s]]]
         .astype(np.float32) * signs[s][ids[site == s]][:, None]
         for s in range(NSITES)])
    Un /= np.linalg.norm(Un, axis=1, keepdims=True)

    mean_ci = np.array([cc[f"{SITES[s]}|mean"][i] for s, i in zip(site, ids)])
    Fp = {s: pf[f"{SITES[s]}|Fp"] for s in range(NSITES)}
    fires_tot = np.array([Fp[s][i].sum() for s, i in zip(site, ids)])
    pos0_share = np.array([Fp[s][i, 0] / max(t, 1)
                           for (s, i), t in zip(zip(site, ids), fires_tot)])

    def top_tokens(x, n=5):
        s, i = site[x], ids[x]
        tids = tt[f"{SITES[s]}|top_ids"][i]
        tci = tt[f"{SITES[s]}|top_ci"][i]
        tot = tt[f"{SITES[s]}|total"][i]
        if tot <= 0:
            return "(no CI in sample)"
        out = []
        for t, c in zip(tids, tci):
            if len(out) >= n or (len(out) >= 3 and c / tot < 0.005):
                break
            out.append("`" + esc(repr(tok.decode([int(t)]))) + "`")
        return " ".join(out)

    def top1(x):
        """Top activating token id, or None if the component never fired."""
        s, i = site[x], ids[x]
        if tt[f"{SITES[s]}|total"][i] <= 0:
            return None
        return int(tt[f"{SITES[s]}|top_ids"][i][0])

    def unemb_align(x):
        """Signed cos(U_x, unemb[top-1 token of x]) under the activation gauge."""
        t = top1(x)
        return None if t is None else float(Un[x] @ unemb[t])

    def name_of(x):
        return f"{SHORT[site[x]]}:{ids[x]}"

    def kind(a, b):
        if site[a] == site[b]:
            return "same-matrix"
        if site[a] // 2 == site[b] // 2:
            return "same-layer"
        return "cross-layer"

    # ---- figures ----------------------------------------------------------
    INK, HUE = "#333333", "#2a6fdb"
    hist = z["hist"]
    edges = np.linspace(-1, 1, 801)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.stairs(np.maximum(hist, 0.5), edges, fill=True, color=HUE, alpha=0.85,
              label="observed")
    rand = n_pairs_tot * random_cos_pdf(centers) * (edges[1] - edges[0])
    ax.plot(centers, np.maximum(rand, 1e-3), color=INK, lw=1.2, ls=":",
            label=f"random directions in d={D}")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.7)   # hide the 0.5 floor drawn in empty bins
    for v, c, lab in ((THR, "#b0413e", f"cluster threshold +{THR}"),
                      (NEG_THR, "#7a5195", f"negative tail < {NEG_THR}")):
        ax.axvline(v, color=c, ls="--", lw=1)
        ax.text(v + 0.01, 1e3, lab, color=c, fontsize=8)
    ax.set_xlabel("cos(U_a, U_b)  (majority-positive-activation gauge)")
    ax.set_ylabel("pair count (log)")
    ax.set_title(f"Signed write-out cosines, all {n_pairs_tot/1e6:.1f}M pooled "
                 f"pairs ({cfg['title']})")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "ucos_hist.png", dpi=150)
    plt.close(fig)

    pm = P[P[:, 2] > THR]
    cnt = np.zeros((NSITES, NSITES), int)
    for a, b in zip(pm[:, 0].astype(int), pm[:, 1].astype(int)):
        sa, sb = sorted((site[a], site[b]))
        cnt[sa, sb] += 1
        if sa != sb:
            cnt[sb, sa] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(np.log10(cnt + 1), cmap="Blues")
    ax.set_xticks(range(NSITES), SHORT, rotation=90, fontsize=8)
    ax.set_yticks(range(NSITES), SHORT, fontsize=8)
    for i in range(NSITES):
        for j in range(NSITES):
            if cnt[i, j]:
                ax.text(j, i, cnt[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if np.log10(cnt[i, j] + 1) > 0.6 * np.log10(cnt.max() + 1) else INK)
    for x in np.arange(1.5, NSITES - 1, 2):
        ax.axhline(x, color="#999999", lw=0.7)
        ax.axvline(x, color="#999999", lw=0.7)
    ax.set_title(f"pairs with cos(U) > {THR} per site pair")
    fig.colorbar(im, ax=ax, shrink=0.75, label="log10(count+1)")
    fig.tight_layout()
    fig.savefig(FIG / "ucos_site_pairs.png", dpi=150)
    plt.close(fig)

    # ---- clusters (positive edges only) -----------------------------------
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge = defaultdict(dict)
    for a, b, c in pm:
        a, b = int(a), int(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        edge[a][b] = edge[b][a] = c
    groups = defaultdict(list)
    for x in range(N):
        groups[find(x)].append(x)
    clusters = sorted((v for v in groups.values() if len(v) > 1),
                      key=lambda c: (-len({site[x] for x in c}), -len(c),
                                     site[c[0]], ids[c[0]]))
    in_cluster = sum(len(c) for c in clusters)

    # ---- unembedding-alignment analyses -----------------------------------
    by_layer = defaultdict(list)
    for c in clusters:
        for x in c:
            a = unemb_align(x)
            if a is not None:
                by_layer[site[x] // 2].append(a)
    layer_med = {l: np.median(v) for l, v in by_layer.items()}

    def mean_dir(members):
        vs = Un[members]        # already consistently gauge-oriented
        m = vs.mean(0)
        return m / np.linalg.norm(m)

    # same trigger token, one pure-L0 cluster + one pure-upper cluster
    by_tok = defaultdict(list)
    for c in clusters:
        toks = Counter(t for t in (top1(x) for x in c) if t is not None)
        if not toks:
            continue
        t, layers = toks.most_common(1)[0][0], {site[x] // 2 for x in c}
        by_tok[t].append((c, layers))
    matched = []
    for t, cl in by_tok.items():
        l0 = [c for c, ly in cl if ly == {0}]
        up = [c for c, ly in cl if 0 not in ly]
        if l0 and up:
            d0, du = mean_dir(max(l0, key=len)), mean_dir(max(up, key=len))
            matched.append((t, len(max(l0, key=len)), len(max(up, key=len)),
                            float(d0 @ unemb[t]), float(du @ unemb[t]),
                            float(d0 @ du)))
    matched.sort(key=lambda r: -(r[1] + r[2]))

    eos_clusters = [c for c in clusters
                    if Counter(t for t in (top1(x) for x in c) if t is not None)
                    .most_common(1)[0][0] == 0 and len(c) >= 5]
    eos_align = np.median([float(Un[x] @ unemb[0])
                           for c in eos_clusters for x in c]) if eos_clusters else np.nan

    # ---- co-CI lookups ----------------------------------------------------
    CI_PATH = HERE / "cache" / f"cluster_ci_signed_{name}.npz"
    kidx, Rk = {}, None
    if CI_PATH.exists():
        cz = np.load(CI_PATH)
        kidx = {(int(s), int(i)): k
                for k, (s, i) in enumerate(zip(cz["site"], cz["ids"]))}
        Tk = float(cz["T"])
        mean_k = cz["S1"] / Tk
        cov = cz["G"] / Tk - np.outer(mean_k, mean_k)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            Rk = cov / np.outer(sd, sd)

    r_cache = {}

    def coci_r(a, b):
        """Pearson r of per-token CI; cross-site via the cluster_ci Gram."""
        k1, k2 = kidx.get((site[a], ids[a])), kidx.get((site[b], ids[b]))
        if Rk is not None and k1 is not None and k2 is not None:
            v = float(Rk[k1, k2])
            return v if np.isfinite(v) else None
        if site[a] != site[b]:
            return None
        s = site[a]
        if s not in r_cache:
            r_cache[s] = cc[f"{SITES[s]}|r"]
        return float(r_cache[s][ids[a], ids[b]])

    def v_cos(a, b):
        """Signed read-side cosine (same gauge), same-matrix pairs only."""
        if site[a] != site[b]:
            return None
        s = site[a]
        V = uv[f"{SITES[s]}|V"]
        va = V[row_of[s][int(ids[a])]].astype(np.float32) * signs[s][ids[a]]
        vb = V[row_of[s][int(ids[b])]].astype(np.float32) * signs[s][ids[b]]
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def cluster_fig(no, order):
        """cos(U) + co-CI r heatmaps over the cluster members (table order)."""
        n = len(order)
        labels = [name_of(x) for x in order]
        Uc = Un[order]
        C = Uc @ Uc.T
        R = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                r = coci_r(order[i], order[j])
                if r is not None:
                    R[i, j] = r
        cell = 0.32
        fig, axes = plt.subplots(
            1, 2, figsize=(2 * n * cell + 3.2, n * cell + 1.6), sharey=True)
        for ax, M, title in ((axes[0], C, "cos(U)"),
                             (axes[1], R, "co-CI r")):
            cm = plt.get_cmap("RdBu_r").copy()
            cm.set_bad("#dddddd")
            im = ax.imshow(M, cmap=cm, vmin=-1, vmax=1)
            ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[0].set_yticks(range(n), labels, fontsize=7)
        fig.tight_layout()
        fig.savefig(CFIG / f"cluster_{no}.png", dpi=150)
        plt.close(fig)
        return f"{relfig}/clusters/cluster_{no}.png"

    # ---- report -----------------------------------------------------------
    pos_hi = [(P[:, 2] > t).sum() for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    neg_hi = [(P[:, 2] < -t).sum() for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    kind_cnt = Counter(kind(int(a), int(b)) for a, b, _ in pm)
    sm = [(int(a), int(b), c) for a, b, c in pm if site[int(a)] == site[int(b)]]
    rs = np.array([coci_r(a, b) for a, b, _ in sm])
    vcs_arr = np.array([v_cos(a, b) for a, b, _ in sm])
    neg = P[P[:, 2] < NEG_THR]
    neg = neg[np.argsort(neg[:, 2])]

    unemb_name = ("wte (tied — the unembedding)" if cfg["tied"]
                  else "lm_head (untied head)")

    L = []
    L.append(f"# {cfg['title']} ({cfg['run']}): signed write-out (U) cosines "
             "across the residual-stream-writing matrices\n")
    L.append("Write-side mirror of the signed read-in report "
             "(`../read-in/`): here the vectors compared are the components' "
             "**write directions** $U_c$ — the rank-one write into the "
             "residual stream is $a_c(x)\\,U_c$ — for the 8 matrices that "
             "write to the stream, `h.<l>.attn.o_proj` and "
             "`h.<l>.mlp.down_proj`. "
             "Sign convention: each component's $(U_c, V_c)$ gauge is flipped "
             "so its input activation $V_c\\cdot x$ is positive on the "
             "majority of tokens where it fires (CI > 0.1) — the interactive "
             "cross-widget convention. Under it, $\\cos(U_a,U_b) > 0$ means "
             "two components write the same stream direction with the same "
             "polarity when active; $< 0$, opposite polarities.\n")

    mt_rows = "\n".join(
        f"| `{esc(repr(tok.decode([t])))}` | {n0} | {nu} | {a0:+.2f} | {au:+.2f} | {c01:+.2f} |"
        for t, n0, nu, a0, au, c01 in matched)
    mt_block = ""
    if matched:
        mt_block = f"""
| token | n(L0 cl.) | n(upper cl.) | cos(L0, unemb) | cos(up, unemb) | cos(L0, up) |
|---|---|---|---|---|---|
{mt_rows}

  (Auto-generated: token = the modal top-1 activating token of the cluster,
  L0/upper cluster = the largest all-layer-0 resp. no-layer-0 cluster with
  that modal token, directions = member means under the activation gauge.)
"""
    eos_line = ""
    if eos_clusters:
        big_eos = max(eos_clusters, key=len)
        eos_line = (f"- **EOS/boundary machinery:** cluster "
                    f"{clusters.index(big_eos) + 1} ({len(big_eos)} comps) has "
                    f"modal top token `<|endoftext|>`; median signed "
                    f"cos(U, unemb[EOS]) over its members = {eos_align:+.2f}.\n")
    L.append(f"""
## Headlines

- **{in_cluster} of {N} pooled components ({100 * in_cluster / N:.0f}%) sit in
  {len(clusters)} clusters** chained at cos > +{THR}
  ({kind_cnt['same-matrix']} same-matrix, {kind_cnt['same-layer']} same-layer
  cross-matrix, {kind_cnt['cross-layer']} cross-layer pairs above +{THR}).
- **The negative tail is substantial: {len(neg)} pairs below {NEG_THR}
  ({neg_hi[3]} below -{THR}), strongest {neg[0, 2] if len(neg) else np.nan:+.2f}** —
  vs {pos_hi[0]} above +{-NEG_THR} ({pos_hi[3]} above +{THR}).
- **Signed unembedding alignment by layer** (clustered members' cos(U,
  unemb[own top token]), median): L0 {layer_med.get(0, np.nan):+.2f} /
  L1 {layer_med.get(1, np.nan):+.2f} / L2 {layer_med.get(2, np.nan):+.2f} /
  L3 {layer_med.get(3, np.nan):+.2f}. (unemb = {unemb_name}.)
{mt_block}{eos_line}- **Same-matrix aligned pairs**: median co-CI
  r = {np.median(rs):.2f}, median signed read cos(V) = {np.median(vcs_arr):+.2f}
  ({(vcs_arr > 0.7).sum()} pairs > +0.7, {(vcs_arr < -0.1).sum()} < -0.1).
""")
    L.extend(NOTES[name])
    L.append(f"""
## Method

All **alive** components (sample mean CI > 1e-6 over the 4,000 cached Pile
rows) of the 8 matrices whose $U$ factor writes to the residual stream —
`h.<l>.attn.o_proj` and `h.<l>.mlp.down_proj`, $l=0..3$ — pooled into one set
of **{N} components**; the reading matrices (q/k/v/c_fc) are the subject of
the read-in reports. For every pair the **signed** write alignment

$$\\cos(U_a, U_b) = \\frac{{U_a \\cdot U_b}}{{\\lVert U_a\\rVert\\,\\lVert U_b\\rVert}},$$

with each component first put in the majority-positive-activation gauge
(sign statistics over the same 2.05M tokens; `comp_signs()`). Unlike the read
side there is no gain-folding question: the write $a_c(x)\\,U_c$ enters the
raw residual stream directly (no norm between $U$ and the stream), so raw $U$
is exactly the write direction. For random unit vectors in $d=768$ the signed
cosine is symmetric around 0 with
$\\mathrm{{std}} = 1/\\sqrt d \\approx {1/np.sqrt(D):.3f}$
(and $E|\\cos| = \\sqrt{{2/\\pi d}} \\approx {BASELINE:.3f}$); the dotted
line in the histogram is this analytic null scaled to the pair count.
Scripts: `hide/compute_signed.py` → `hide/cache/ucos_signed_{name}.npz`,
`hide/report_signed.py` (this report); decomposition {cfg['run']} of target
`{cfg['target']}`.

## Distribution

![histogram]({relfig}/ucos_hist.png)

| threshold | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| pairs above +thr | {pos_hi[0]} | {pos_hi[1]} | {pos_hi[2]} | {pos_hi[3]} | {pos_hi[4]} | {pos_hi[5]} |
| pairs below -thr | {neg_hi[0]} | {neg_hi[1]} | {neg_hi[2]} | {neg_hi[3]} | {neg_hi[4]} | {neg_hi[5]} |

({n_pairs_tot / 1e6:.1f}M pairs total; max {P[:, 2].max():+.3f}, min
{P[:, 2].min():+.3f}.)

Of the {len(pm)} pairs above +{THR}: **{kind_cnt['same-matrix']} same-matrix,
{kind_cnt['same-layer']} same-layer cross-matrix, {kind_cnt['cross-layer']}
cross-layer**.

![site pairs]({relfig}/ucos_site_pairs.png)

## Clusters (chaining at cos > +{THR})

Connected components of the cos > +{THR} graph: **{len(clusters)} clusters
with ≥ 2 members, covering {in_cluster} of {N} components**. Clusters are
ordered by the number of distinct matrices involved (ties by size); members
by matrix, then mean CI descending.
Tables below: all clusters with ≥ 5 members in full, smaller ones compacted.
Per member: sample mean CI, share of CI>0.1 fires at position 0, signed
alignment of U with the member's own top token's unembedding ("unemb align"),
top activating tokens (share of summed CI). Below each table: member × member
heatmaps of cos(U) and co-CI r (cross-site values from
`hide/cache/cluster_ci_signed_{name}.npz`), members in table order, both on
the same RdBu scale.
""")

    BIG = 5
    for ci_, c in enumerate(clusters):
        if len(c) < BIG:
            break
        ecos = [edge[a][b] for a in c for b in edge[a] if b in c and a < b]
        csites = sorted({SHORT[site[x]] for x in c})
        L.append(f"\n### Cluster {ci_ + 1} — n = {len(c)} "
                 f"({', '.join(csites)}); edge cos {min(ecos):+.2f}–{max(ecos):+.2f}\n")
        L.append("| matrix | component | mean CI | pos-0 fires | unemb align | top tokens |")
        L.append("|---|---|---|---|---|---|")
        order = sorted(c, key=lambda x: (site[x], -mean_ci[x]))
        prev_site = None
        for x in order:
            p0 = f"{100 * pos0_share[x]:.0f}%" if fires_tot[x] >= 20 else "–"
            ea = unemb_align(x)
            ea = "–" if ea is None else f"{ea:+.2f}"
            mat = SHORT[site[x]] if site[x] != prev_site else ""
            prev_site = site[x]
            L.append(f"| {mat} | {ids[x]} | {mean_ci[x]:.1e} | {p0} | {ea} | {top_tokens(x)} |")
        L.append(f"\n![cluster {ci_ + 1}]({cluster_fig(ci_ + 1, order)})")

    def cluster_kind(c):
        if len({site[x] for x in c}) == 1:
            return "same-matrix"
        if len({site[x] // 2 for x in c}) == 1:
            return "same-layer"
        return "cross-layer"

    KIND_ORDER = {"cross-layer": 0, "same-layer": 1, "same-matrix": 2}
    small = [c for c in clusters if 2 < len(c) < BIG]
    pairs2 = [c for c in clusters if len(c) == 2]
    L.append(f"\n### Smaller clusters (3–{BIG - 1} members, {len(small)} of them) — grouped by kind\n")
    L.append("| members | kind | top tokens (first member) |")
    L.append("|---|---|---|")
    for c in sorted(small, key=lambda c: KIND_ORDER[cluster_kind(c)]):
        mem = " ".join(name_of(x) for x in sorted(c, key=lambda x: (site[x], ids[x])))
        best = max(c, key=lambda x: mean_ci[x])
        L.append(f"| {mem} | {cluster_kind(c)} | {top_tokens(best, 4)} |")

    L.append(f"\n### Pairs ({len(pairs2)} two-member clusters) — the {min(30, len(pairs2))} "
             "highest-cos shown, grouped by kind\n")
    L.append("| pair | cos U | kind | co-CI r | cos V | top tokens (a / b) |")
    L.append("|---|---|---|---|---|---|")
    p2 = sorted(pairs2, key=lambda c: -max(edge[c[0]].get(c[1], 0), edge[c[1]].get(c[0], 0)))
    p2 = sorted(p2[:30], key=lambda c: KIND_ORDER[kind(c[0], c[1])])
    for c in p2:
        a, b = sorted(c, key=lambda x: (site[x], ids[x]))
        cv = edge[a][b]
        r = coci_r(a, b)
        vcs = v_cos(a, b)
        L.append(f"| {name_of(a)} ↔ {name_of(b)} | {cv:+.2f} | {kind(a, b)} | "
                 f"{'' if r is None else f'{r:.2f}'} | {'' if vcs is None else f'{vcs:+.2f}'} | "
                 f"{top_tokens(a, 3)} / {top_tokens(b, 3)} |")

    # ---- negative tail ----------------------------------------------------
    NSHOW = min(25, len(neg))
    L.append(f"""
## Negative tail (cos < {NEG_THR})

{len(neg)} pairs write the same stream direction with **opposite** polarity at
cos < {NEG_THR} (vs {(P[:, 2] > -NEG_THR).sum()} positive pairs above
+{-NEG_THR}); the strongest {NSHOW} (co-CI r from the signed cluster-CI Gram —
its members include every component in a pair below {NEG_THR}):
""")
    L.append("| pair | cos U | kind | co-CI r | cos V | top tokens (a / b) |")
    L.append("|---|---|---|---|---|---|")
    for a, b, cv in neg[:NSHOW]:
        a, b = int(a), int(b)
        if site[b] < site[a] or (site[a] == site[b] and ids[b] < ids[a]):
            a, b = b, a
        r = coci_r(a, b)
        vcs = v_cos(a, b)
        L.append(f"| {name_of(a)} ↔ {name_of(b)} | {cv:+.2f} | {kind(a, b)} | "
                 f"{'' if r is None else f'{r:.2f}'} | {'' if vcs is None else f'{vcs:+.2f}'} | "
                 f"{top_tokens(a, 3)} / {top_tokens(b, 3)} |")

    xs = [(int(a), int(b)) for a, b, c in pm if site[int(a)] != site[int(b)]]
    rx = np.array([r for r in (coci_r(a, b) for a, b in xs) if r is not None])
    neg_rs = np.array([r for r in (coci_r(int(a), int(b)) for a, b, _ in neg)
                       if r is not None])
    L.append(f"""
## Do aligned writers co-fire / co-read?

For the {len(sm)} **same-matrix** pairs above +{THR}: median co-CI
r = {np.median(rs):.2f} (quartiles {np.percentile(rs, 25):.2f}–{np.percentile(rs, 75):.2f};
{(rs > 0.9).sum()} pairs > 0.9, {(rs < 0.1).sum()} pairs < 0.1) and median
signed read cos(V) = {np.median(vcs_arr):+.2f} ({(vcs_arr > 0.7).sum()} pairs
> +0.7, {(np.abs(vcs_arr) < 0.1).sum()} with |cos V| < 0.1,
{(vcs_arr < -0.1).sum()} < -0.1). The {len(rx)} **cross-site** pairs above
+{THR} have median co-CI r = {np.median(rx) if len(rx) else np.nan:.2f}
(quartiles {np.percentile(rx, 25) if len(rx) else np.nan:.2f}–{np.percentile(rx, 75) if len(rx) else np.nan:.2f};
{(rx > 0.9).sum() if len(rx) else 0} > 0.9, {(rx < 0.1).sum() if len(rx) else 0} < 0.1;
from the cross-site CI Gram of all clustered components, Modal job
`hide/cluster_ci_signed_modal.py` → `hide/cache/cluster_ci_signed_{name}.npz`).
The {len(neg_rs)} negative-tail pairs with a defined r have median co-CI
r = {np.median(neg_rs) if len(neg_rs) else np.nan:.2f}
({(neg_rs < 0).sum() if len(neg_rs) else 0} of them negative).
V factors of different sites live in different input spaces, so no cross-site
cos(V) is defined.
""")

    # ---- within-matrix distributions --------------------------------------
    bins = np.linspace(-1, 1, 201)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(4, 2, figsize=(9, 9.5), sharex=True)
    for s in range(NSITES):
        ax = axes.flat[s]
        X = Un[site == s]
        n = len(X)
        nps = n * (n - 1) // 2
        vals_s = (X @ X.T)[np.triu_indices(n, k=1)]
        h = np.histogram(vals_s, bins=bins)[0]
        ax.stairs(np.maximum(h, 0.5), bins, fill=True, color=HUE, alpha=0.85)
        rnd = nps * random_cos_pdf(ctr) * (bins[1] - bins[0])
        ax.plot(ctr, np.maximum(rnd, 1e-3), color=INK, lw=1.0, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.7)
        ax.set_title(f"{SHORT[s]}  (n = {n}, {nps / 1e3:.0f}k pairs)",
                     fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Within-matrix signed cos(U), alive components "
                 f"({cfg['title']}); dotted = random directions in d={D}",
                 fontsize=11)
    fig.supxlabel("cos(U_a, U_b)", fontsize=10)
    fig.supylabel("pair count (log)", fontsize=10)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(FIG / "ucos_within_matrix.png", dpi=150)
    plt.close(fig)

    L.append(f"""
## Within-matrix cosine distributions

One panel per matrix: the distribution of signed cos(U) over all pairs of
alive components *within* that matrix (log count; dotted line = the analytic
random-directions null in $d = {D}$, scaled to the panel's pair count).

![within-matrix distributions]({relfig}/ucos_within_matrix.png)
""")

    (ROOT / "cos-sim" / "write-out" / cfg["report"]).write_text(
        "\n".join(L), encoding="utf-8")
    print(f"{name}: report written -> {cfg['report']}; {len(clusters)} clusters, "
          f"{in_cluster} comps in clusters, {len(neg)} negative-tail pairs")


for nm in sys.argv[1:] or ["newA", "C"]:
    render(nm)
