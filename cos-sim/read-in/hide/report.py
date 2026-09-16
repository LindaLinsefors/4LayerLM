"""Render the UNSIGNED new-A read-in-cosine report + figures from cache/vcos.npz.

ARCHIVED 2026-09-16: the rendered report + its figures live in
archive/read-in-newA-unsigned-2026-09-16/ (cos-sim/read-in/newA_report.md is
now the SIGNED report); rerunning this script writes
cos-sim/read-in/newA_unsigned_report.md instead so nothing is clobbered.

Clusters the |cos(V)| > THR graph over the pooled read-side alive components
(union-find chaining), annotates members with sample mean CI, pos-0 fire share
and top activating tokens, and writes the report.  Run with PYTHONUTF8=1.
Inputs: cache/vcos.npz (compute.py), coci-heatmaps/hide/cache/{uv_newA.npz,
coci_newA.npz, pos_fires_newA.npz}, mean-ci-widget/hide/cache/top_tokens_newA.npz.
"""
import sys
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
from load import load_tokenizer

CH = ROOT / "coci-heatmaps" / "hide" / "cache"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")]
SHORT = [f"L{l}{m}" for l in range(4) for m in ("q", "k", "v", "c_fc")]
THR = 0.7
D = 768
BASELINE = np.sqrt(2 / (np.pi * D))

z = np.load(HERE / "cache" / "vcos.npz")
site, ids = z["site"], z["ids"]
N = len(site)
P = z["pairs_raw"]
Pf = z["pairs_fold"]
tok = load_tokenizer("pile_4l")
tt = np.load(ROOT / "mean-ci-widget" / "hide" / "cache" / "top_tokens_newA.npz")
cc = np.load(CH / "coci_newA.npz")
pf = np.load(CH / "pos_fires_newA.npz")
uv = np.load(CH / "uv_newA.npz")
alive = np.load(CH / "cross_alive.npz")
# position within the site's alive list (for co-CI / U lookups)
alive_pos = {}
for s, st_ in enumerate(SITES):
    a = alive[f"newA|{st_}"]
    alive_pos[s] = {int(c): k for k, c in enumerate(a)}

st_ = safe_open(ROOT / "prev_paper" / "models" / "pile_4layer"
                / "target_model_t-9d2b8f02" / "model_step_99999.safetensors",
                framework="np")
wte = st_.get_tensor("wte.weight").astype(np.float32)
wte /= np.linalg.norm(wte, axis=1, keepdims=True)
Vn = np.concatenate([uv[f"{s}|V"].astype(np.float32) for s in SITES])
Vn /= np.linalg.norm(Vn, axis=1, keepdims=True)

mean_ci = np.array([cc[f"{SITES[s]}|mean"][i] for s, i in zip(site, ids)])
Fp = {s: pf[f"{SITES[s]}|Fp"] for s in range(16)}
fires_tot = np.array([Fp[s][i].sum() for s, i in zip(site, ids)])
pos0_share = np.array([Fp[s][i, 0] / max(t, 1)
                       for (s, i), t in zip(zip(site, ids), fires_tot)])


def esc(t):
    # tokens go inside code spans: literal except table pipes (GFM special case)
    return t.replace("|", "\\|").replace("`", "'")


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
    """|cos(V_x, wte[top-1 token of x])|."""
    t = top1(x)
    return None if t is None else abs(float(Vn[x] @ wte[t]))


def name(x):
    return f"{SHORT[site[x]]}:{ids[x]}"


def kind(a, b):
    if site[a] == site[b]:
        return "same-matrix"
    if site[a] // 4 == site[b] // 4:
        return "same-layer"
    return "cross-layer"


# ---- figures --------------------------------------------------------------
INK, HUE = "#333333", "#2a6fdb"
hist = z["hist_raw"]
edges = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.stairs(np.maximum(hist, 0.5), edges, fill=True, color=HUE, alpha=0.85)
ax.set_yscale("log")
ax.axvline(BASELINE, color="#888888", ls="--", lw=1)
ax.text(BASELINE + 0.01, hist.max() / 3,
        f"random baseline √(2/πd) = {BASELINE:.3f}", color="#666666", fontsize=8)
ax.axvline(THR, color="#b0413e", ls="--", lw=1)
ax.text(THR + 0.01, 1e3, f"cluster threshold {THR}", color="#b0413e", fontsize=8)
ax.set_xlabel("|cos(V_a, V_b)|")
ax.set_ylabel("pair count (log)")
ax.set_title("Read-in cosine distribution, all 37.5M pooled pairs (new A)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eeeeee", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "vcos_hist.png", dpi=150)
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
ax.set_title(f"pairs with |cos(V)| > {THR} per site pair")
fig.colorbar(im, ax=ax, shrink=0.75, label="log10(count+1)")
fig.tight_layout()
fig.savefig(FIG / "vcos_site_pairs.png", dpi=150)
plt.close(fig)

# ---- clusters -------------------------------------------------------------
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


def cnum(short, cid):
    """1-based report number of the cluster containing component short:cid."""
    s = SHORT.index(short)
    for k, c in enumerate(clusters):
        if any(site[x] == s and ids[x] == cid for x in c):
            return k + 1
    return None

# ---- embedding-alignment analyses -----------------------------------------
by_layer = defaultdict(list)
for c in clusters:
    for x in c:
        a = emb_align(x)
        if a is not None:
            by_layer[site[x] // 4].append(a)
layer_med = {l: np.median(v) for l, v in by_layer.items()}


def mean_dir(members):
    vs = Vn[members]
    vs = vs * np.sign(vs @ vs[0] + 1e-9)[:, None]   # fix sign gauge
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
                        abs(d0 @ wte[t]), abs(du @ wte[t]), abs(d0 @ du)))
matched.sort(key=lambda r: -(r[1] + r[2]))

eos_clusters = [c for c in clusters
                if Counter(t for t in (top1(x) for x in c) if t is not None)
                .most_common(1)[0][0] == 0 and len(c) >= 5]
eos_align = np.median([abs(Vn[x] @ wte[0])
                       for c in eos_clusters for x in c]) if eos_clusters else np.nan

# ---- co-CI lookups --------------------------------------------------------
# cross-site co-CI for all clustered components (Modal job cluster_ci_modal.py)
CI_PATH = HERE / "cache" / "cluster_ci.npz"
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
    if site[a] != site[b]:
        return None
    U = uv[f"{SITES[site[a]]}|U"]
    ua = U[alive_pos[site[a]][int(ids[a])]].astype(np.float32)
    ub = U[alive_pos[site[b]][int(ids[b])]].astype(np.float32)
    return abs(float(ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub))))


CFIG = FIG / "clusters"
CFIG.mkdir(exist_ok=True)


def cluster_fig(no, order):
    """|cos(V)| + co-CI r heatmaps over the cluster members (table order)."""
    n = len(order)
    labels = [name(x) for x in order]
    Vc = Vn[order]
    C = np.abs(Vc @ Vc.T)
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            r = coci_r(order[i], order[j])
            if r is not None:
                R[i, j] = r
    cell = 0.32
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * n * cell + 3.2, n * cell + 1.6), sharey=True)
    for ax, M, cmap, vmin, title in (
            (axes[0], C, "Blues", 0, "|cos(V)|"),
            (axes[1], R, "RdBu_r", -1, "co-CI r")):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#dddddd")
        im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=1)
        ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_yticks(range(n), labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(CFIG / f"cluster_{no}.png", dpi=150)
    plt.close(fig)
    return f"hide/figures/clusters/cluster_{no}.png"


# ---- report ---------------------------------------------------------------
vals_hi = [(P[:, 2] > t).sum() for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
kind_cnt = Counter(kind(int(a), int(b)) for a, b, _ in pm)
# same-matrix high-cos pairs: are they behavioral duplicates?
sm = [(int(a), int(b), c) for a, b, c in pm if site[int(a)] == site[int(b)]]
rs = np.array([coci_r(a, b) for a, b, _ in sm])
ucs_arr = np.array([u_cos(a, b) for a, b, _ in sm])

L = []
L.append("# New A: read-in (V) cosine similarity across the residual-stream-reading matrices\n")
mt_rows = "\n".join(
    f"| `{esc(repr(tok.decode([t])))}` | {n0} | {nu} | {a0:.2f} | {au:.2f} | {c01:.2f} |"
    for t, n0, nu, a0, au, c01 in matched)
L.append(f"""
## Headlines

- **The high-cosine tail is token-identity machinery.** {in_cluster} of {N}
  pooled components ({100 * in_cluster / N:.0f}%) sit in {len(clusters)}
  clusters at |cos(V)| > {THR}, and almost every cluster is "all the
  components, across matrix types and layers, that fire on token/class X"
  (`\\`, `-`, `,`, ` the`, ` and`, ` (`, `:`, `;`, newline, whitespace runs,
  digits, single letters, Cyrillic, ` de`, possessives, …). Most alignment is
  **between** sites, not within one matrix ({kind_cnt['same-layer']}
  same-layer cross-matrix + {kind_cnt['cross-layer']} cross-layer vs
  {kind_cnt['same-matrix']} same-matrix pairs above {THR}): q, k, v and c_fc
  components in several layers read the *same* direction of the residual
  stream.
- **Each token has two orthogonal read directions: the embedding (read by
  layer 0) and a recoded identity direction (read by layers 1–3).** Clustered
  L0 members' read-ins align with their trigger token's embedding (median
  |cos(V, wte)| = {layer_med.get(0, np.nan):.2f}), upper-layer members' do not
  (medians {layer_med.get(1, np.nan):.2f} / {layer_med.get(2, np.nan):.2f} /
  {layer_med.get(3, np.nan):.2f} for L1/L2/L3) — yet upper-layer members of
  one cluster agree with each other at 0.7+. Accordingly the |cos| > {THR}
  graph splits into an L0 block and an L1–L3 block (see the site-pair figure):
  for several tokens the *same* trigger has one pure-L0 cluster reading
  $\\approx$ wte and one disjoint L1–L3 cluster reading a direction orthogonal
  to it. Layer 0 (or early layer 1) evidently **re-encodes token identity into
  fresh directions**, and everything downstream reads the re-encoding:

| token | n(L0 cl.) | n(upper cl.) | \\|cos(L0, wte)\\| | \\|cos(up, wte)\\| | \\|cos(L0, up)\\| |
|---|---|---|---|---|---|
{mt_rows}

  (Auto-generated: token = the modal top-1 activating token of the cluster,
  L0/upper cluster = the largest all-layer-0 resp. no-layer-0 cluster with
  that modal token, directions = gauge-fixed member means. The few rows with
  low |cos(L0, wte)| — `'\\n'`, `'s'`, `' de'`, `'Ð'` — are multi-token
  *class* detectors (letters, Spanish function words, Cyrillic bytes,
  whitespace variants), where alignment with any single member token's
  embedding is diluted; the actual L0 newline readers (in mixed cluster
  {cnum('L0k', 128)}) align with wte['\\n'] at ~0.67.)

  The small L0↔L3c_fc off-diagonal block in the site-pair figure is the one
  partial exception: a handful of L3 c_fc readers align with the
  L0/embedding directions of frequent tokens (e.g. cluster
  {cnum('L3c_fc', 858)}'s L3c_fc:858 with the newline readers, cluster
  {cnum('L3c_fc', 2156)}'s L3c_fc:2156 with the ' of' readers) — the raw
  embedding is still present in the layer-3 residual stream and is
  occasionally read there directly.

- **The pos-0/EOS sink machinery shares a read-in too.** The big
  EOS/boundary cluster (cluster
  {clusters.index(max(eos_clusters, key=len)) + 1 if eos_clusters else '?'}:
  {len(max(eos_clusters, key=len)) if eos_clusters else 0}
  comps in L2/L3, most with majority pos-0 fire share) reads a common
  direction that is *not* the EOS embedding (median |cos(V, wte[EOS])| =
  {eos_align:.2f}) — consistent with it reading the shared massive-vector /
  boundary direction of the endoftext-pos0 sub-project.
- **Aligned readers are not component duplicates.** Same-matrix pairs above
  {THR} co-fire moderately (median co-CI r = {np.median(rs):.2f}) but write
  near-orthogonally (median |cos(U)| = {np.median(ucs_arr):.2f}) — the
  decomposition splits one input feature into
  several components with different outputs, rather than duplicating whole
  components.
""")
L.append(f"""
## Method

All **alive** components (sample mean CI > 1e-6 over the 4,000 cached Pile rows,
the standard proxy for the new decompositions) of the 16 matrices whose $V$
factor reads the residual stream — `h.<l>.attn.{{q,k,v}}_proj` (input
$\\mathrm{{rms}}_1(h)$) and `h.<l>.mlp.c_fc` (input $\\mathrm{{rms}}_2(h)$), $l=0..3$ —
pooled into one set of **{N} components**; `o_proj` (reads attention-head
outputs) and `down_proj` (reads the MLP hidden layer) are excluded. For every
pair we compute the read-in alignment

$$|\\cos(V_a, V_b)| = \\frac{{|V_a \\cdot V_b|}}{{\\lVert V_a\\rVert\\,\\lVert V_b\\rVert}}$$

(absolute value because component sign is gauge). Random baseline for unit
vectors in $d=768$: $\\sqrt{{2/\\pi d}} \\approx {BASELINE:.3f}$. Scripts:
`hide/compute.py` (pooled cosine matrix → `hide/cache/vcos.npz`),
`hide/report.py` (this report). V factors from the alive-only dump
`coci-heatmaps/hide/cache/uv_newA.npz`.

**Gain-folding check:** the effective read direction on the unit-normalized
stream is $g \\odot V$ (site RMSNorm gain $g$, since $V\\cdot(\\hat x \\odot g)
= (g\\odot V)\\cdot\\hat x$), and $g$ differs between sites. Folding the gains in
changes essentially nothing — pairs above 0.4/0.7/0.9:
{(P[:,2]>0.4).sum()}/{(P[:,2]>0.7).sum()}/{(P[:,2]>0.9).sum()} raw vs
{(Pf[:,2]>0.4).sum()}/{(Pf[:,2]>0.7).sum()}/{(Pf[:,2]>0.9).sum()} gain-folded —
so raw $V$ cosines are used throughout.

## Distribution

![histogram](hide/figures/vcos_hist.png)

The bulk is random-baseline-like (median |cos| = 0.023); a far tail of
genuinely aligned pairs sticks out:

| threshold | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| pairs above | {vals_hi[0]} | {vals_hi[1]} | {vals_hi[2]} | {vals_hi[3]} | {vals_hi[4]} | {vals_hi[5]} |

(37.5M pairs total; max |cos| = {P[:,2].max():.3f}.)

Of the {len(pm)} pairs above {THR}: **{kind_cnt['same-matrix']} same-matrix,
{kind_cnt['same-layer']} same-layer cross-matrix, {kind_cnt['cross-layer']}
cross-layer** — most read-in alignment is *between* sites, not within one
matrix.

![site pairs](hide/figures/vcos_site_pairs.png)

## Clusters (chaining at |cos| > {THR})

Connected components of the |cos| > {THR} graph: **{len(clusters)} clusters
with ≥ 2 members, covering {in_cluster} of {N} components**. Clusters are
ordered by the number of distinct matrices involved (ties by size); members
by matrix, then mean CI descending.
Tables below: all clusters with ≥ 5 members in full, smaller ones compacted.
Per member: sample mean CI, share of CI>0.1 fires at position 0, alignment of
V with the member's own top token's embedding ("emb align"), top activating
tokens (share of summed CI). Within-cluster |cos| ranges are over the edges
present in the graph (> {THR}). Below each table: member × member heatmaps of
|cos(V)| (read-in alignment) and co-CI r (per-token CI Pearson correlation,
cross-site values from `hide/cache/cluster_ci.npz`), members in table order.
""")

BIG = 5
for ci_, c in enumerate(clusters):
    if len(c) < BIG:
        break
    ecos = [edge[a][b] for a in c for b in edge[a] if b in c and a < b]
    csites = sorted({SHORT[site[x]] for x in c})
    L.append(f"\n### Cluster {ci_ + 1} — n = {len(c)} "
             f"({', '.join(csites)}); edge |cos| {min(ecos):.2f}–{max(ecos):.2f}\n")
    L.append("| matrix | component | mean CI | pos-0 fires | emb align | top tokens |")
    L.append("|---|---|---|---|---|---|")
    order = sorted(c, key=lambda x: (site[x], -mean_ci[x]))
    prev_site = None
    for x in order:
        p0 = f"{100 * pos0_share[x]:.0f}%" if fires_tot[x] >= 20 else "–"
        ea = emb_align(x)
        ea = "–" if ea is None else f"{ea:.2f}"
        mat = SHORT[site[x]] if site[x] != prev_site else ""
        prev_site = site[x]
        L.append(f"| {mat} | {ids[x]} | {mean_ci[x]:.1e} | {p0} | {ea} | {top_tokens(x)} |")
    L.append(f"\n![cluster {ci_ + 1}]({cluster_fig(ci_ + 1, order)})")

def cluster_kind(c):
    """Same classification as pair kind, for a whole cluster."""
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
    mem = " ".join(name(x) for x in sorted(c, key=lambda x: (site[x], ids[x])))
    best = max(c, key=lambda x: mean_ci[x])
    L.append(f"| {mem} | {cluster_kind(c)} | {top_tokens(best, 4)} |")

L.append(f"\n### Pairs ({len(pairs2)} two-member clusters) — the {min(30, len(pairs2))} "
         "highest-|cos| shown, grouped by kind\n")
L.append("| pair | \\|cos V\\| | kind | co-CI r | \\|cos U\\| | top tokens (a / b) |")
L.append("|---|---|---|---|---|---|")
p2 = sorted(pairs2, key=lambda c: -max(edge[c[0]].get(c[1], 0), edge[c[1]].get(c[0], 0)))
p2 = sorted(p2[:30], key=lambda c: KIND_ORDER[kind(c[0], c[1])])
for c in p2:
    a, b = sorted(c, key=lambda x: (site[x], ids[x]))
    cv = edge[a][b]
    r = coci_r(a, b)
    ucs = u_cos(a, b)
    L.append(f"| {name(a)} ↔ {name(b)} | {cv:.2f} | {kind(a, b)} | "
             f"{'' if r is None else f'{r:.2f}'} | {'' if ucs is None else f'{ucs:.2f}'} | "
             f"{top_tokens(a, 3)} / {top_tokens(b, 3)} |")

xs = [(int(a), int(b)) for a, b, c in pm if site[int(a)] != site[int(b)]]
rx = np.array([r for r in (coci_r(a, b) for a, b in xs) if r is not None])
L.append(f"""
## Do aligned readers co-fire / co-write?

For the {len(sm)} **same-matrix** pairs above {THR}: median co-CI
r = {np.median(rs):.2f} (quartiles {np.percentile(rs,25):.2f}–{np.percentile(rs,75):.2f};
{(rs>0.9).sum()} pairs > 0.9, {(rs<0.1).sum()} pairs < 0.1) and median
write-side |cos(U)| = {np.median(ucs_arr):.2f} ({(ucs_arr>0.7).sum()} pairs > 0.7,
{(ucs_arr<0.1).sum()} < 0.1). The {len(rx)} **cross-site** pairs co-fire
similarly: median co-CI r = {np.median(rx):.2f} (quartiles
{np.percentile(rx,25):.2f}–{np.percentile(rx,75):.2f}; {(rx>0.9).sum()} > 0.9,
{(rx<0.1).sum()} < 0.1; from the cross-site CI Gram of all clustered
components, Modal job `hide/cluster_ci_modal.py` → `hide/cache/cluster_ci.npz`).
U factors of different sites live in different output spaces, so no cross-site
|cos(U)| is defined.
""")

(ROOT / "cos-sim" / "read-in" / "newA_unsigned_report.md").write_text(
    "\n".join(L), encoding="utf-8")
print(f"report written; {len(clusters)} clusters, {in_cluster} comps in clusters")
