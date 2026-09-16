"""Render the SIGNED read-in-cosine reports: cos-sim/read-in/newA_report.md
(named newA_signed_report.md until 2026-09-16, when the unsigned report was
archived) and cos-sim/read-in/C_report.md (+ figures) from
cache/vcos_signed_{decomp}.npz.

Same analysis as report.py (the unsigned new-A report) but cosines keep their
sign under the majority-positive-activation gauge (see compute_signed.py):
clusters chain at cos > +0.7, and the negative tail (cos < -0.4) gets its own
section.  Run with PYTHONUTF8=1.

Usage: python cos-sim/read-in/hide/report_signed.py [newA C]   (default: both)
Inputs per decomposition: cache/vcos_signed_*.npz (compute_signed.py),
cache/cluster_ci_signed_*.npz (cluster_ci_signed_modal.py), the uv/act_signs/
coci/top_tokens/pos-fires caches listed in DECOMPS, and the target-model
safetensors (wte for embedding alignment).
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

SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")]
SHORT = [f"L{l}{m}" for l in range(4) for m in ("q", "k", "v", "c_fc")]
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
        unsigned="../../archive/read-in-newA-unsigned-2026-09-16/newA_report.md"),
    "C": dict(
        title="Sink decomposition C", run="p-d60af588", target="t-87f91319",
        report="C_report.md",
        uv=CD / "uv_C.npz", acts=CD / "act_signs_C.npz",
        coci=CH / "coci_C.npz",
        tt=ROOT / "mean-ci-widget" / "hide" / "cache" / "top_tokens_C.npz",
        pf=ROOT / "sink-models" / "hide" / "cache" / "pos_fires_sink_C.npz",
        st=ROOT / "sink-models" / "pretrain_cache" / "spd-t-87f91319"
        / "model_step_100000.safetensors",
        unsigned=None),
}

# hand-written decomposition-specific commentary, inserted into Headlines
NOTES = {
    "newA": ["""
- **The sign adds no surprises to the clusters — and that is the finding.**
  Of the 1,097 pairs with |cos| > 0.7 in the unsigned report, only 8 are
  negative under the gauge; the cluster structure is unchanged (289 clusters,
  977 vs 981 members — the 4 components that entered only via negative edges
  drop out). Co-firing components that read the same stream direction
  essentially always read it with the same polarity.
- **Layer 0 reads *toward* the token embedding.** The signed L0 median
  (+0.66) equals the unsigned one: under the majority-positive-activation
  gauge, every L0 token-reader cluster sits on the $+\\mathrm{wte}$ side —
  positive activation means "token present", never the inverted read. Upper
  layers are orthogonal (medians +0.03/+0.01/+0.00), **not anti-aligned**:
  the re-encoding of token identity is a rotation to fresh directions, not a
  sign flip.
- **The negative tail has one dominant motif: L3 c_fc components reading the
  $-\\mathrm{wte}$ side of a function word they tend to precede.** 18 of the
  20 strongest negative pairs put an L0 reader of token X against an L3 c_fc
  component whose read-in is $\\approx -\\mathrm{wte}[X]$ and whose top
  activating tokens are words that typically *precede* X — L3c_fc:917
  (fires on ` used`, ` have`) vs the ` to` readers, L3c_fc:2213 (` used`,
  ` available`) vs ` for`, L3c_fc:230 vs ` that` — all with co-CI r ≈ 0
  (median −0.00 over the tail): not anti-firing partners but readers of the
  opposite side of the same embedding axis, plausibly next-token-prediction
  machinery. The one true opposite-polarity same-trigger pair is
  L3v:145 ↔ L3c_fc:534 (both fire on `V`, cos −0.74, co-CI r 0.36).
"""],
    "C": ["""
- **C shares read-ins an order of magnitude less than new A.** 91 pairs above
  +0.7 vs new A's 1,089 (on a *larger* pooled set, 10,612 vs 8,657); max cos
  0.89 vs 0.95; 1% of components clustered vs 11%; largest cluster n = 7 vs
  n = 22. The pervasive "every reader of token X shares one direction"
  machinery of new A is only weakly present in the sink decomposition.
- **No EOS/boundary cluster.** New A's biggest cross-matrix cluster was the
  22-comp EOS/pos-0 sink read direction; C has no cluster with modal token
  `<|endoftext|>` at all — consistent with the sink-models finding that the
  built-in attention sinks remove the emergent boundary machinery these
  components read.
- **The L0-vs-upper split survives, but the re-encoding is less complete.**
  L0 members read toward the token embedding (median +0.70); upper-layer
  members are only weakly aligned — yet at L1 the median is +0.19 (new A:
  +0.03), and several matched tokens keep cos(up, wte) ≈ +0.15..0.26: in
  these targets the raw embedding remains partially readable (and read) in
  upper layers.
- **The negative tail's clearest motif is ± feature splits inside v_proj.**
  L2v:334 ↔ L2v:573 (cos V −0.65, co-CI r −0.41, cos U +0.45) and
  L1v:144 ↔ L1v:634 (−0.60, −0.42, +0.59): opposite-polarity reads of one
  axis, anti-co-firing, *similar* write directions — one feature axis split
  into a component per sign.
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

    z = np.load(HERE / "cache" / f"vcos_signed_{name}.npz")
    site, ids = z["site"], z["ids"]
    N = len(site)
    P = z["pairs_raw"]
    Pf = z["pairs_fold"]
    n_pairs_tot = N * (N - 1) // 2
    tok = load_tokenizer("pile_4l")
    tt = np.load(cfg["tt"])
    cc = np.load(cfg["coci"])
    pf = np.load(cfg["pf"])
    uv = np.load(cfg["uv"])
    acts = np.load(cfg["acts"])
    signs = {s: comp_signs(acts, SITES[s]) for s in range(16)}

    # row lookup into the uv dump: newA holds only alive rows (in the pooled
    # id order), C holds all C rows in id order
    row_of = {}
    for s in range(16):
        a = ids[site == s]
        full = len(uv[f"{SITES[s]}|V"]) == cc[f"{SITES[s]}|mean"].shape[0]
        row_of[s] = ({int(c): int(c) for c in a} if full
                     else {int(c): k for k, c in enumerate(a)})

    st_ = safe_open(cfg["st"], framework="np")
    wte = st_.get_tensor("wte.weight").astype(np.float32)
    wte /= np.linalg.norm(wte, axis=1, keepdims=True)
    # gauge-fixed unit read-ins, pooled order (matches compute_signed.py)
    Vn = np.concatenate(
        [uv[f"{SITES[s]}|V"][[row_of[s][int(c)] for c in ids[site == s]]]
         .astype(np.float32) * signs[s][ids[site == s]][:, None]
         for s in range(16)])
    Vn /= np.linalg.norm(Vn, axis=1, keepdims=True)

    mean_ci = np.array([cc[f"{SITES[s]}|mean"][i] for s, i in zip(site, ids)])
    Fp = {s: pf[f"{SITES[s]}|Fp"] for s in range(16)}
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

    def emb_align(x):
        """Signed cos(V_x, wte[top-1 token of x]) under the activation gauge."""
        t = top1(x)
        return None if t is None else float(Vn[x] @ wte[t])

    def name_of(x):
        return f"{SHORT[site[x]]}:{ids[x]}"

    def kind(a, b):
        if site[a] == site[b]:
            return "same-matrix"
        if site[a] // 4 == site[b] // 4:
            return "same-layer"
        return "cross-layer"

    # ---- figures ----------------------------------------------------------
    INK, HUE = "#333333", "#2a6fdb"
    hist = z["hist_raw"]
    edges = np.linspace(-1, 1, 801)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rand = n_pairs_tot * random_cos_pdf(centers) * (edges[1] - edges[0])
    for logy, fname in ((True, "vcos_hist.png"), (False, "vcos_hist_linear.png")):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.stairs(np.maximum(hist, 0.5) if logy else hist, edges, fill=True,
                  color=HUE, alpha=0.85, label="observed")
        ax.plot(centers, np.maximum(rand, 1e-3) if logy else rand,
                color=INK, lw=1.2, ls=":",
                label=f"random directions in d={D}")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.7)   # hide the 0.5 floor drawn in empty bins
        for v, c, lab in ((THR, "#b0413e", f"cluster threshold +{THR}"),
                          (NEG_THR, "#7a5195", f"negative tail < {NEG_THR}")):
            ax.axvline(v, color=c, ls="--", lw=1)
            ax.text(v + 0.01, 0.8, lab, color=c, fontsize=8,
                    transform=ax.get_xaxis_transform())
        ax.set_xlabel("cos(V_a, V_b)  (majority-positive-activation gauge)")
        ax.set_ylabel("pair count" + (" (log)" if logy else ""))
        ax.set_title(f"Signed read-in cosines, all {n_pairs_tot/1e6:.1f}M pooled "
                     f"pairs ({cfg['title']})")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    pm = P[P[:, 2] > THR]
    cnt = np.zeros((16, 16), int)
    for a, b in zip(pm[:, 0].astype(int), pm[:, 1].astype(int)):
        sa, sb = sorted((site[a], site[b]))
        cnt[sa, sb] += 1
        if sa != sb:
            cnt[sb, sa] += 1
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(np.log10(cnt + 1), cmap="Blues")
    ax.set_xticks(range(16), SHORT, rotation=90, fontsize=8)
    ax.set_yticks(range(16), SHORT, fontsize=8)
    for i in range(16):
        for j in range(16):
            if cnt[i, j]:
                ax.text(j, i, cnt[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if np.log10(cnt[i, j] + 1) > 0.6 * np.log10(cnt.max() + 1) else INK)
    for x in (3.5, 7.5, 11.5):
        ax.axhline(x, color="#999999", lw=0.7)
        ax.axvline(x, color="#999999", lw=0.7)
    ax.set_title(f"pairs with cos(V) > {THR} per site pair")
    fig.colorbar(im, ax=ax, shrink=0.75, label="log10(count+1)")
    fig.tight_layout()
    fig.savefig(FIG / "vcos_site_pairs.png", dpi=150)
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

    # ---- embedding-alignment analyses -------------------------------------
    by_layer = defaultdict(list)
    for c in clusters:
        for x in c:
            a = emb_align(x)
            if a is not None:
                by_layer[site[x] // 4].append(a)
    layer_med = {l: np.median(v) for l, v in by_layer.items()}

    def mean_dir(members):
        vs = Vn[members]        # already consistently gauge-oriented
        m = vs.mean(0)
        return m / np.linalg.norm(m)

    # same trigger token, one pure-L0 cluster + one pure-upper cluster
    by_tok = defaultdict(list)
    for c in clusters:
        toks = Counter(t for t in (top1(x) for x in c) if t is not None)
        if not toks:
            continue
        t, layers = toks.most_common(1)[0][0], {site[x] // 4 for x in c}
        by_tok[t].append((c, layers))
    matched = []
    for t, cl in by_tok.items():
        l0 = [c for c, ly in cl if ly == {0}]
        up = [c for c, ly in cl if 0 not in ly]
        if l0 and up:
            d0, du = mean_dir(max(l0, key=len)), mean_dir(max(up, key=len))
            matched.append((t, len(max(l0, key=len)), len(max(up, key=len)),
                            float(d0 @ wte[t]), float(du @ wte[t]),
                            float(d0 @ du)))
    matched.sort(key=lambda r: -(r[1] + r[2]))

    eos_clusters = [c for c in clusters
                    if Counter(t for t in (top1(x) for x in c) if t is not None)
                    .most_common(1)[0][0] == 0 and len(c) >= 5]
    eos_align = np.median([float(Vn[x] @ wte[0])
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

    def u_cos(a, b):
        """Signed write-side cosine (same gauge), same-matrix pairs only."""
        if site[a] != site[b]:
            return None
        s = site[a]
        U = uv[f"{SITES[s]}|U"]
        ua = U[row_of[s][int(ids[a])]].astype(np.float32) * signs[s][ids[a]]
        ub = U[row_of[s][int(ids[b])]].astype(np.float32) * signs[s][ids[b]]
        return float(ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub)))

    def cluster_fig(no, order):
        """cos(V) + co-CI r heatmaps over the cluster members (table order)."""
        n = len(order)
        labels = [name_of(x) for x in order]
        Vc = Vn[order]
        C = Vc @ Vc.T
        R = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                r = coci_r(order[i], order[j])
                if r is not None:
                    R[i, j] = r
        cell = 0.32
        fig, axes = plt.subplots(
            1, 2, figsize=(2 * n * cell + 3.2, n * cell + 1.6), sharey=True)
        for ax, M, title in ((axes[0], C, "cos(V)"),
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
    ucs_arr = np.array([u_cos(a, b) for a, b, _ in sm])
    neg = P[P[:, 2] < NEG_THR]
    neg = neg[np.argsort(neg[:, 2])]

    L = []
    L.append(f"# {cfg['title']} ({cfg['run']}): signed read-in (V) cosines "
             "across the residual-stream-reading matrices\n")
    if cfg["unsigned"]:
        L.append("Signed-cosine version of the original unsigned |cos| report "
                 f"(same pooled set, same clustering threshold), archived at "
                 f"[{Path(cfg['unsigned']).name}]({cfg['unsigned']}). ")
    L.append("Sign convention: each component's $(U_c, V_c)$ gauge is flipped "
             "so its input activation $V_c\\cdot x$ is positive on the "
             "majority of tokens where it fires (CI > 0.1) — the interactive "
             "cross-widget convention. Under it, $\\cos(V_a,V_b) > 0$ means "
             "two components read the same stream direction with the same "
             "polarity; $< 0$, opposite polarities.\n")

    mt_rows = "\n".join(
        f"| `{esc(repr(tok.decode([t])))}` | {n0} | {nu} | {a0:+.2f} | {au:+.2f} | {c01:+.2f} |"
        for t, n0, nu, a0, au, c01 in matched)
    mt_block = ""
    if matched:
        mt_block = f"""
| token | n(L0 cl.) | n(upper cl.) | cos(L0, wte) | cos(up, wte) | cos(L0, up) |
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
                    f"cos(V, wte[EOS]) over its members = {eos_align:+.2f}.\n")
    L.append(f"""
## Headlines

- **{in_cluster} of {N} pooled components ({100 * in_cluster / N:.0f}%) sit in
  {len(clusters)} clusters** chained at cos > +{THR}
  ({kind_cnt['same-matrix']} same-matrix, {kind_cnt['same-layer']} same-layer
  cross-matrix, {kind_cnt['cross-layer']} cross-layer pairs above +{THR}).
- **The negative tail is thin: {len(neg)} pairs below {NEG_THR}, strongest
  {neg[0, 2] if len(neg) else np.nan:+.2f}** — under the activation gauge,
  strong read-in alignment is essentially always same-polarity; there is no
  sizable population of components reading the same feature with opposite
  sign at |cos| > {THR}.
- **Signed embedding alignment by layer** (clustered members' cos(V, wte[own
  top token]), median): L0 {layer_med.get(0, np.nan):+.2f} /
  L1 {layer_med.get(1, np.nan):+.2f} / L2 {layer_med.get(2, np.nan):+.2f} /
  L3 {layer_med.get(3, np.nan):+.2f}.
{mt_block}{eos_line}- **Same-matrix aligned pairs**: median co-CI
  r = {np.median(rs):.2f}, median signed write cos(U) = {np.median(ucs_arr):+.2f}
  ({(ucs_arr > 0.7).sum()} pairs > +0.7, {(ucs_arr < -0.1).sum()} < -0.1).
""")
    L.extend(NOTES[name])
    L.append(f"""
## Method

All **alive** components (sample mean CI > 1e-6 over the 4,000 cached Pile
rows) of the 16 matrices whose $V$ factor reads the residual stream —
`h.<l>.attn.{{q,k,v}}_proj` (input $\\mathrm{{rms}}_1(h)$) and `h.<l>.mlp.c_fc`
(input $\\mathrm{{rms}}_2(h)$), $l=0..3$ — pooled into one set of
**{N} components**; `o_proj` and `down_proj` are excluded. For every pair the
**signed** read-in alignment

$$\\cos(V_a, V_b) = \\frac{{V_a \\cdot V_b}}{{\\lVert V_a\\rVert\\,\\lVert V_b\\rVert}},$$

with each component first put in the majority-positive-activation gauge
(sign statistics over the same 2.05M tokens; `comp_signs()`). For random unit
vectors in $d=768$ the signed cosine is symmetric around 0 with
$\\mathrm{{std}} = 1/\\sqrt d \\approx {1/np.sqrt(D):.3f}$
(and $E|\\cos| = \\sqrt{{2/\\pi d}} \\approx {BASELINE:.3f}$). Scripts:
`hide/compute_signed.py` → `hide/cache/vcos_signed_{name}.npz`,
`hide/report_signed.py` (this report); decomposition {cfg['run']} of target
`{cfg['target']}`.

**Gain-folding check:** folding the site RMSNorm gains into $V$ (effective
read on the unit stream is $g \\odot V$) changes essentially nothing — pairs
above +0.4/+0.7 resp. below -0.4:
{(P[:, 2] > 0.4).sum()}/{(P[:, 2] > 0.7).sum()}/{(P[:, 2] < -0.4).sum()} raw vs
{(Pf[:, 2] > 0.4).sum()}/{(Pf[:, 2] > 0.7).sum()}/{(Pf[:, 2] < -0.4).sum()}
gain-folded — so raw $V$ cosines are used throughout.

## Distribution

![histogram]({relfig}/vcos_hist.png)

| threshold | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| pairs above +thr | {pos_hi[0]} | {pos_hi[1]} | {pos_hi[2]} | {pos_hi[3]} | {pos_hi[4]} | {pos_hi[5]} |
| pairs below -thr | {neg_hi[0]} | {neg_hi[1]} | {neg_hi[2]} | {neg_hi[3]} | {neg_hi[4]} | {neg_hi[5]} |

({n_pairs_tot / 1e6:.1f}M pairs total; max {P[:, 2].max():+.3f}, min
{P[:, 2].min():+.3f}.)

Of the {len(pm)} pairs above +{THR}: **{kind_cnt['same-matrix']} same-matrix,
{kind_cnt['same-layer']} same-layer cross-matrix, {kind_cnt['cross-layer']}
cross-layer**.

![site pairs]({relfig}/vcos_site_pairs.png)

## Clusters (chaining at cos > +{THR})

Connected components of the cos > +{THR} graph: **{len(clusters)} clusters
with ≥ 2 members, covering {in_cluster} of {N} components**. Clusters are
ordered by the number of distinct matrices involved (ties by size); members
by matrix, then mean CI descending.
Tables below: all clusters with ≥ 5 members in full, smaller ones compacted.
Per member: sample mean CI, share of CI>0.1 fires at position 0, signed
alignment of V with the member's own top token's embedding ("emb align"),
top activating tokens (share of summed CI). Below each table: member × member
heatmaps of cos(V) and co-CI r (cross-site values from
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
        L.append("| matrix | component | mean CI | pos-0 fires | emb align | top tokens |")
        L.append("|---|---|---|---|---|---|")
        order = sorted(c, key=lambda x: (site[x], -mean_ci[x]))
        prev_site = None
        for x in order:
            p0 = f"{100 * pos0_share[x]:.0f}%" if fires_tot[x] >= 20 else "–"
            ea = emb_align(x)
            ea = "–" if ea is None else f"{ea:+.2f}"
            mat = SHORT[site[x]] if site[x] != prev_site else ""
            prev_site = site[x]
            L.append(f"| {mat} | {ids[x]} | {mean_ci[x]:.1e} | {p0} | {ea} | {top_tokens(x)} |")
        L.append(f"\n![cluster {ci_ + 1}]({cluster_fig(ci_ + 1, order)})")

    def cluster_kind(c):
        if len({site[x] for x in c}) == 1:
            return "same-matrix"
        if len({site[x] // 4 for x in c}) == 1:
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
    L.append("| pair | cos V | kind | co-CI r | cos U | top tokens (a / b) |")
    L.append("|---|---|---|---|---|---|")
    p2 = sorted(pairs2, key=lambda c: -max(edge[c[0]].get(c[1], 0), edge[c[1]].get(c[0], 0)))
    p2 = sorted(p2[:30], key=lambda c: KIND_ORDER[kind(c[0], c[1])])
    for c in p2:
        a, b = sorted(c, key=lambda x: (site[x], ids[x]))
        cv = edge[a][b]
        r = coci_r(a, b)
        ucs = u_cos(a, b)
        L.append(f"| {name_of(a)} ↔ {name_of(b)} | {cv:+.2f} | {kind(a, b)} | "
                 f"{'' if r is None else f'{r:.2f}'} | {'' if ucs is None else f'{ucs:+.2f}'} | "
                 f"{top_tokens(a, 3)} / {top_tokens(b, 3)} |")

    # ---- negative tail ----------------------------------------------------
    NSHOW = min(20, len(neg))
    L.append(f"""
## Negative tail (cos < {NEG_THR})

{len(neg)} pairs read the same stream direction with **opposite** polarity at
cos < {NEG_THR} (vs {(P[:, 2] > -NEG_THR).sum()} positive pairs above
+{-NEG_THR}); the strongest {NSHOW} (co-CI r from the signed cluster-CI Gram —
its members include every component in a pair below {NEG_THR}):
""")
    L.append("| pair | cos V | kind | co-CI r | cos U | top tokens (a / b) |")
    L.append("|---|---|---|---|---|---|")
    for a, b, cv in neg[:NSHOW]:
        a, b = int(a), int(b)
        if site[b] < site[a] or (site[a] == site[b] and ids[b] < ids[a]):
            a, b = b, a
        r = coci_r(a, b)
        ucs = u_cos(a, b)
        L.append(f"| {name_of(a)} ↔ {name_of(b)} | {cv:+.2f} | {kind(a, b)} | "
                 f"{'' if r is None else f'{r:.2f}'} | {'' if ucs is None else f'{ucs:+.2f}'} | "
                 f"{top_tokens(a, 3)} / {top_tokens(b, 3)} |")

    xs = [(int(a), int(b)) for a, b, c in pm if site[int(a)] != site[int(b)]]
    rx = np.array([r for r in (coci_r(a, b) for a, b in xs) if r is not None])
    neg_rs = np.array([r for r in (coci_r(int(a), int(b)) for a, b, _ in neg)
                       if r is not None])
    L.append(f"""
## Do aligned readers co-fire / co-write?

For the {len(sm)} **same-matrix** pairs above +{THR}: median co-CI
r = {np.median(rs):.2f} (quartiles {np.percentile(rs, 25):.2f}–{np.percentile(rs, 75):.2f};
{(rs > 0.9).sum()} pairs > 0.9, {(rs < 0.1).sum()} pairs < 0.1) and median
signed write cos(U) = {np.median(ucs_arr):+.2f} ({(ucs_arr > 0.7).sum()} pairs
> +0.7, {(np.abs(ucs_arr) < 0.1).sum()} with |cos U| < 0.1,
{(ucs_arr < -0.1).sum()} < -0.1). The {len(rx)} **cross-site** pairs co-fire
similarly: median co-CI r = {np.median(rx):.2f} (quartiles
{np.percentile(rx, 25):.2f}–{np.percentile(rx, 75):.2f}; {(rx > 0.9).sum()} > 0.9,
{(rx < 0.1).sum()} < 0.1; from the cross-site CI Gram of all clustered
components, Modal job `hide/cluster_ci_signed_modal.py` →
`hide/cache/cluster_ci_signed_{name}.npz`). The {len(neg_rs)} negative-tail
pairs with a defined r have median co-CI r = {np.median(neg_rs) if len(neg_rs) else np.nan:.2f}
({(neg_rs < 0).sum() if len(neg_rs) else 0} of them negative).
U factors of different sites live in different output spaces, so no cross-site
cos(U) is defined.
""")

    # ---- within-matrix distributions --------------------------------------
    bins = np.linspace(-1, 1, 201)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    for logy, fname in ((True, "vcos_within_matrix.png"),
                        (False, "vcos_within_matrix_linear.png")):
        fig, axes = plt.subplots(4, 4, figsize=(13, 9.5), sharex=True)
        for s in range(16):
            ax = axes.flat[s]
            X = Vn[site == s]
            n = len(X)
            nps = n * (n - 1) // 2
            vals_s = (X @ X.T)[np.triu_indices(n, k=1)]
            h = np.histogram(vals_s, bins=bins)[0]
            ax.stairs(np.maximum(h, 0.5) if logy else h, bins, fill=True,
                      color=HUE, alpha=0.85)
            rnd = nps * random_cos_pdf(ctr) * (bins[1] - bins[0])
            ax.plot(ctr, np.maximum(rnd, 1e-3) if logy else rnd,
                    color=INK, lw=1.0, ls=":")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.7)
            ax.set_title(f"{SHORT[s]}  (n = {n}, {nps / 1e3:.0f}k pairs)",
                         fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#eeeeee", lw=0.6)
            ax.set_axisbelow(True)
        fig.suptitle(f"Within-matrix signed cos(V), alive components "
                     f"({cfg['title']}); dotted = random directions in d={D}",
                     fontsize=11)
        fig.supxlabel("cos(V_a, V_b)", fontsize=10)
        fig.supylabel("pair count" + (" (log)" if logy else ""), fontsize=10)
        fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
        fig.savefig(FIG / fname, dpi=150)
        plt.close(fig)

    L.append(f"""
## Within-matrix cosine distributions

One panel per matrix: the distribution of signed cos(V) over all pairs of
alive components *within* that matrix (log count; dotted line = the analytic
random-directions null in $d = {D}$, scaled to the panel's pair count).

![within-matrix distributions]({relfig}/vcos_within_matrix.png)

## Linear-scale versions

The same distribution plots with a linear y axis (the log plots emphasize
the tails; these show where the actual mass sits).

![histogram linear]({relfig}/vcos_hist_linear.png)

![within-matrix distributions linear]({relfig}/vcos_within_matrix_linear.png)
""")

    (ROOT / "cos-sim" / "read-in" / cfg["report"]).write_text(
        "\n".join(L), encoding="utf-8")
    print(f"{name}: report written -> {cfg['report']}; {len(clusters)} clusters, "
          f"{in_cluster} comps in clusters, {len(neg)} negative-tail pairs")


for nm in sys.argv[1:] or ["newA", "C"]:
    render(nm)
