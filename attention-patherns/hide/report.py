"""Render figures + report.md for the sink45 layer-0 attention analysis.

Reads hide/cache/l0_attn_stats.npz (built by compute.py) and writes, one level up:
  l0_sink_mass_vs_position.png, l0_offset_profile.png, l0_token_sink_hist.png,
  report.md
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT = HERE.parent
sys.path.insert(0, str(ROOT))
import load  # noqa: E402

D = np.load(HERE / "cache" / "l0_attn_stats.npz")
tok = load.load_tokenizer("pile_4l")
NH, T = 6, 512
# fixed head <-> color mapping (Okabe-Ito, CVD-safe), same in every figure
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]

sink_pos, key0_pos, neff_pos = D["sink_pos"], D["key0_pos"], D["neff_pos"]
offset = D["offset"]                     # mean mass at distance d over valid queries
sink_logits = D["sink_logits"]
n_rows, window = int(D["n_rows"]), int(D["window"])

# per-query offset weight: mean over ALL queries (missing distances count as 0),
# so that sum_d pq[d] = mean content mass = 1 - mean sink mass
counts_d = n_rows * (T - np.arange(T))
pq = offset * counts_d / (n_rows * T)

mean_sink = sink_pos[:, 64:].mean(1)
mean_neff = neff_pos[:, 64:].mean(1)
content = pq.sum(1)
share8 = pq[:, :9].sum(1) / content
share32 = pq[:, :33].sum(1) / content


def tokstr(i):
    s = tok.decode([int(i)])
    s = s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("|", "\\|")
    return f"`'{s}'`"


# ---------------- figures ----------------
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for pane, (ax, xs) in enumerate(zip(axes, [np.arange(T), np.arange(64)])):
    for h in range(NH):
        ax.plot(xs, sink_pos[h, xs], color=COLORS[h], lw=1.6, label=f"head {h}")
    ax.set_xlabel("query position $i$ in the 512-token chunk")
    ax.set_ylim(0, 1.02)
    ax.set_title("full chunk" if pane == 0 else "first 64 positions (zoom)")
axes[0].set_ylabel("mean sink mass  $\\langle A_{i,\\mathrm{sink}}\\rangle$")
axes[0].legend(loc="center right", fontsize=8, framealpha=0.9)
for h in range(NH):  # direct labels at the right edge of the full panel
    axes[0].annotate(f"{h}", (T + 4, sink_pos[h, -8:].mean()), color=COLORS[h],
                     fontsize=8, va="center", annotation_clip=False)
fig.suptitle("Sink seed 45, layer 0: attention mass on the built-in sink slot vs query position "
             f"({n_rows} Pile rows)")
fig.tight_layout()
fig.savefig(OUT / "l0_sink_mass_vs_position.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ds = np.arange(1, T)
for h in range(NH):
    ax.plot(ds, offset[h, 1:], color=COLORS[h], lw=1.6, label=f"head {h}")
    ax.scatter([0.7], [offset[h, 0]], color=COLORS[h], s=14, zorder=3)  # d=0 marker
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("query$-$key distance $d$   (markers at left edge: $d=0$, self)")
ax.set_ylabel("mean attention mass  $\\langle A_{i,i-d}\\rangle$")
ax.set_title("Layer 0: mean attention mass vs distance (log–log)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "l0_offset_profile.png", bbox_inches="tight")
plt.close(fig)

# same x-axis, but mean pre-softmax logit; reference: each head's sink logit
offset_logit = D["offset_logit"]
fig, ax = plt.subplots(figsize=(7, 4.5))
for h in range(NH):
    ax.plot(ds, offset_logit[h, 1:], color=COLORS[h], lw=1.6, label=f"head {h}")
    ax.scatter([0.7], [offset_logit[h, 0]], color=COLORS[h], s=14, zorder=3)
    ax.axhline(sink_logits[h], color=COLORS[h], lw=0.9, ls=":")
ax.set_xscale("log")
ax.set_xlabel("query$-$key distance $d$   (markers at left edge: $d=0$, self)")
ax.set_ylabel("mean attention logit  $\\langle q_i\\cdot k_{i-d}\\rangle/\\sqrt{128}$")
ax.set_title("Layer 0: mean attention logit vs distance (dotted: sink logit $s_h$)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "l0_offset_logits.png", bbox_inches="tight")
plt.close(fig)

# per-query sink-mass histograms (spread over individual queries, positions >= 64)
sink_hist = D["sink_hist"]                    # (6, 100), bins on [0,1]
n_bins = sink_hist.shape[1]
edges = np.linspace(0, 1, n_bins + 1)
centers = (edges[:-1] + edges[1:]) / 2
frac = sink_hist / sink_hist.sum(1, keepdims=True)
for scale, fname in [("log", "l0_sink_mass_query_hist.png"),
                     ("linear", "l0_sink_mass_query_hist_linear.png")]:
    # linear y: per-head y scale (peak heights differ a lot); log y: shared scale
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, sharey=(scale == "log"))
    for h, ax in enumerate(axes.flat):
        ax.bar(centers, frac[h], width=1 / n_bins, color=COLORS[h])
        ax.axvline(mean_sink[h], color="k", lw=1, ls="--")
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(f"head {h}", color=COLORS[h])
    for ax in axes[1]:
        ax.set_xlabel("sink mass $A_{i,\\mathrm{sink}}$ of the query")
    axes[0, 0].set_ylabel("fraction of queries"); axes[1, 0].set_ylabel("fraction of queries")
    fig.suptitle("Per-query sink mass distribution, all queries at positions $\\geq$ 64 "
                 f"({n_rows} Pile rows, {scale} y; dashed: mean)")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)

# tail fractions from the histogram (bin resolution 0.01)
def frac_above(h, x):
    return frac[h, edges[:-1] >= x].sum()

def frac_below(h, x):
    return frac[h, edges[1:] <= x].sum()

qhist_rows = "\n".join(
    f"| {h} | {frac_below(h, 0.1):.3f} | {frac_below(h, 0.5):.3f} | "
    f"{frac_above(h, 0.9):.3f} | {frac_above(h, 0.98):.3f} |"
    for h in range(NH))

# per-token mean sink histograms
MIN = 200
qtok_cnt, sink_qtok = D["qtok_cnt"], D["sink_qtok"]
sel = np.where(qtok_cnt >= MIN)[0]
ms = sink_qtok[:, sel] / qtok_cnt[sel]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
for h, ax in enumerate(axes.flat):
    ax.hist(ms[h], bins=np.linspace(0, 1, 41), color=COLORS[h])
    ax.set_title(f"head {h}", color=COLORS[h])
    ax.axvline(mean_sink[h], color="k", lw=1, ls="--")
for ax in axes[1]:
    ax.set_xlabel("mean sink mass of the query token")
axes[0, 0].set_ylabel("token ids"); axes[1, 0].set_ylabel("token ids")
fig.suptitle(f"Per-query-token mean sink mass, the {len(sel)} token ids with $\\geq$ {MIN} "
             "occurrences at positions $\\geq$ 64 (dashed: all-token mean)")
fig.tight_layout()
fig.savefig(OUT / "l0_token_sink_hist.png", bbox_inches="tight")
plt.close(fig)

# OV compensation figure (from ov_compute.py)
OV = np.load(HERE / "cache" / "ov_stats.npz")
wh = OV["wnorm_hist"]
lo, hi = float(OV["log_lo"]), float(OV["log_hi"])
wedges = 10 ** np.linspace(lo, hi, wh.shape[1] + 1)
wcent = np.sqrt(wedges[:-1] * wedges[1:])
cw, cc = OV["cmass_wsum"], OV["cmass_cnt"]
ccent = (np.arange(20) + 0.5) / 20

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for h in range(NH):
    axes[0].stairs(wh[h] / wh[h].sum(), wedges, color=COLORS[h], lw=1.6, label=f"head {h}")
    m = np.where(cc[h] >= 50, cw[h] / np.maximum(cc[h], 1), np.nan)
    axes[1].plot(ccent, m, "o-", ms=3.5, color=COLORS[h], lw=1.6)
axes[0].set_xscale("log")
axes[0].set_xlabel("residual write norm $\\|w_i^h\\|$ of the query")
axes[0].set_ylabel("fraction of queries")
axes[0].set_title("distribution of per-query write norms")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("content mass $c_i = 1 - A_{i,\\mathrm{sink}}$")
axes[1].set_ylabel("mean write norm $\\|w_i^h\\|$")
axes[1].set_title("write norm vs content mass (bins with $\\geq$ 50 queries)")
axes[1].axhline(float(OV["emb_norm_mean"]), color="gray", lw=1, ls=":",
                label="mean $\\|$embedding$\\|$")
axes[1].legend(fontsize=8)
fig.suptitle("Layer 0 OV: what the heads actually write into the residual stream "
             f"({int(OV['n_rows'])} Pile rows, queries at positions ≥ 64)")
fig.tight_layout()
fig.savefig(OUT / "l0_ov_write_norms.png", bbox_inches="tight")
plt.close(fig)

def wq(h, p):  # write-norm quantile from the log-histogram
    cdf = wh[h].cumsum() / wh[h].sum()
    return wcent[np.searchsorted(cdf, p)]

ov_rows = "\n".join(
    f"| {h} | {OV['sv'][h, 0]:.2f} | {np.sqrt((OV['sv'][h] ** 2).sum()):.2f} | "
    f"{OV['vnorm_mean'][h]:.2f} | {content[h]:.2f} | {wq(h, 0.5):.2f} | "
    f"{OV['wnorm_mean'][h]:.2f} | {wq(h, 0.99):.2f} | "
    f"{cw[h, 10] / cc[h, 10]:.2f} |"
    for h in range(NH))

# ---------------- tables ----------------
recv_ktok, ktok_cnt = D["recv_ktok"], D["ktok_cnt"]
selk = np.where(ktok_cnt >= MIN)[0]
mr = recv_ktok[:, selk] / ktok_cnt[selk]

def sink_table(ids):
    lines = ["| token | count | " + " | ".join(f"h{h}" for h in range(NH)) + " | mean |",
             "|---|---:|" + "---:|" * (NH + 1)]
    for i in ids:
        j = np.where(sel == i)[0][0]
        row = " | ".join(f"{ms[h, j]:.2f}" for h in range(NH))
        lines.append(f"| {tokstr(i)} | {qtok_cnt[i]:.0f} | {row} | {ms[:, j].mean():.2f} |")
    return "\n".join(lines)

def recv_table(ids):
    lines = ["| token | count | " + " | ".join(f"h{h}" for h in range(NH)) + " | mean |",
             "|---|---:|" + "---:|" * (NH + 1)]
    for i in ids:
        j = np.where(selk == i)[0][0]
        row = " | ".join(f"{mr[h, j]:.3f}" for h in range(NH))
        lines.append(f"| {tokstr(i)} | {ktok_cnt[i]:.0f} | {row} | {mr[:, j].mean():.3f} |")
    return "\n".join(lines)

# ---------------- per-head sections ----------------
def simple_table(ids, vals, cnts, vfmt, vname):
    lines = [f"| token | count | {vname} |", "|---|---:|---:|"]
    for i, v in zip(ids, vals):
        lines.append(f"| {tokstr(i)} | {cnts[i]:.0f} | {v:{vfmt}} |")
    return "\n".join(lines)

HEAD_NOTES = {
    0: "Wakes mostly on **mid-word fragments and code/markup glue** (`'ing'`, `'in'`, `'](#'`, "
       "`'=\"'`, `'/'`) — consistent with a bigram/word-completion head; receivers are EOS and "
       "opening brackets.",
    1: "The context head: sink mass is near zero for almost everything except EOS; its "
       "content-side selectivity therefore shows in the *receivers* — EOS towers over "
       "everything (0.215), then comment/math openers.",
    2: "A previous-token head that wakes on **quotes, braces and reference markers** "
       "(`'\"'`, `'=\"'`, `'ref'`, `'{'`); receivers are code-line starts (`'){'`, `'if'`) "
       "and paragraph breaks.",
    3: "The broad medium-range head. Strongest content queries are **tab (0.09 sink!) and "
       "code/markup structure** (`'<'`, `' {'`, `' ='`, `'_{'`); receivers are statement "
       "ends/separators (`');'`, `'\",'`, `'**'`, `';'`) plus EOS and prose linkers "
       "(`' between'`, `' who'`).",
    4: "Wakes on **`' to'` and newlines** (plus sentence/clause punctuation) — "
       "infinitive/line-start constructions that need an antecedent; EOS and opening "
       "delimiters receive.",
    5: "A previous-token head keyed to **hyphens, `' of'`, colons and indentation** — "
       "compound-word and list/key-value structure; `'){'` is its one outsize receiver.",
}

# per-distance quantiles (d <= 32) from the cached histograms, for error bars
mass_dhist, logit_dhist = D["mass_dhist"], D["logit_dhist"]
D_MAX = mass_dhist.shape[1] - 1
mass_bin_edges = 10 ** np.linspace(-8, 0, mass_dhist.shape[2] + 1)
logit_bin_edges = np.linspace(-30, 15, logit_dhist.shape[2] + 1)

def dist_quantiles(hist, edges, ps):
    """hist (n_d, n_bins) -> array (len(ps), n_d) of quantiles (bin upper edges)."""
    cdf = hist.cumsum(1) / hist.sum(1, keepdims=True)
    return np.stack([edges[1:][np.argmax(cdf >= p, axis=1)] for p in ps])

def add_quantile_bars(ax, xs, mean, hist, edges, color):
    """Mean line + thick p25-p75 bars + thin p5-p95 whiskers (quantile segments,
    NOT centered on the mean -- the mass distributions are heavily skewed)."""
    q5, q25, q75, q95 = dist_quantiles(hist[xs], edges, [0.05, 0.25, 0.75, 0.95])
    ax.vlines(xs, q5, q95, color=color, lw=0.8, alpha=0.45)
    ax.vlines(xs, q25, q75, color=color, lw=2.2, alpha=0.8)
    ax.plot(xs, mean[xs], "o-", ms=3.5, color=color, lw=1.6, zorder=3)

head_order = np.argsort(mean_sink)
perhead_md = []
for h in head_order:
    ol = np.argsort(ms[h])
    orc = np.argsort(mr[h])
    fig, axes2 = plt.subplots(1, 2, figsize=(11, 3.6))
    axes2[0].plot(np.arange(1, T), offset[h, 1:], color=COLORS[h], lw=1.6)
    axes2[0].scatter([0.7], [offset[h, 0]], color=COLORS[h], s=14, zorder=3)
    axes2[0].set_xscale("log"); axes2[0].set_yscale("log")
    axes2[0].set_xlabel("token distance $d$   (marker at left edge: $d=0$, self)")
    axes2[0].set_ylabel("mean attention mass $\\langle A_{i,i-d}\\rangle$")
    axes2[0].set_title("log–log, full range")
    add_quantile_bars(axes2[1], np.arange(D_MAX + 1), offset[h], mass_dhist[h],
                      mass_bin_edges, COLORS[h])
    axes2[1].set_xlabel("token distance $d$")
    axes2[1].set_title("linear, $d \\leq 32$ (bars p25–p75, whiskers p5–p95)")
    fig.suptitle(f"head {h}: mean attention mass vs token distance")
    fig.tight_layout()
    fig.savefig(OUT / f"l0_offset_h{h}.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes2 = plt.subplots(1, 2, figsize=(11, 3.6))
    axes2[0].plot(np.arange(1, T), offset_logit[h, 1:], color=COLORS[h], lw=1.6)
    axes2[0].set_xscale("log")
    axes2[0].scatter([0.7], [offset_logit[h, 0]], color=COLORS[h], s=14, zorder=3)
    axes2[0].set_ylabel("mean attention logit")
    axes2[0].set_title("log x, full range")
    add_quantile_bars(axes2[1], np.arange(D_MAX + 1), offset_logit[h], logit_dhist[h],
                      logit_bin_edges, COLORS[h])
    axes2[1].set_title("linear, $d \\leq 32$ (bars p25–p75, whiskers p5–p95)")
    for ax in axes2:
        ax.axhline(sink_logits[h], color="k", lw=1, ls="--")
        ax.set_xlabel("token distance $d$")
    axes2[1].annotate(f"sink logit $s_{h}$ = {sink_logits[h]:.2f}",
                      (32, sink_logits[h]), ha="right", va="bottom", fontsize=8)
    fig.suptitle(f"head {h}: mean attention logit vs token distance (dashed: sink logit)")
    fig.tight_layout()
    fig.savefig(OUT / f"l0_offset_logits_h{h}.png", bbox_inches="tight")
    plt.close(fig)

    perhead_md.append(f"""### Head {h} — mean sink mass {mean_sink[h]:.2f}

{HEAD_NOTES[h]}

![head {h} distance profile](l0_offset_h{h}.png)

![head {h} logit distance profile](l0_offset_logits_h{h}.png)

**Query tokens with the lowest sink mass** (most content-attending):

{simple_table(sel[ol[:15]], ms[h][ol[:15]], qtok_cnt, ".2f", "sink mass")}

**Query tokens with the highest sink mass:**

{simple_table(sel[ol[-10:]][::-1], ms[h][ol[-10:]][::-1], qtok_cnt, ".2f", "sink mass")}

**Top receiving tokens** (mean mass from the following 64 queries):

{simple_table(selk[orc[-15:]][::-1], mr[h][orc[-15:]][::-1], ktok_cnt, ".3f", "received mass")}
""")
perhead_md = "\n".join(perhead_md)

agg = ms.mean(0)
o = np.argsort(agg)
low_ids, high_ids = sel[o[:20]], sel[o[-15:]][::-1]
aggr = mr.mean(0)
recv_ids = selk[np.argsort(aggr)[-20:]][::-1]

head_rows = "\n".join(
    f"| {h} | {sink_logits[h]:.2f} | {mean_sink[h]:.3f} | {content[h]:.3f} | "
    f"{mean_neff[h]:.1f} | {share8[h]:.2f} | {share32[h]:.2f} |"
    for h in range(NH))

eos_recv = " / ".join(f"{recv_ktok[h, 0] / ktok_cnt[0]:.3f}" for h in range(NH))
eos_sink = " / ".join(f"{sink_qtok[h, 0] / qtok_cnt[0]:.2f}" for h in range(NH))

report = f"""# Layer-0 attention patterns of the sink seed 45 model

**Model:** sink seed 45 = `t-87f91319` (4L Pile model with untied head and one learned
zero-value attention-sink logit per head), loaded with `load.load_sink(45)` — i.e. with the
**corrected RoPE spectrum** (see `sink-models/rope_report.md`), so these attention patterns
are trustworthy at all ranges.

**Data / method:** the first {n_rows} cached Pile rows (`context-loss/hide/cache/pile_rows.pt`),
truncated to the 512-token context. Layer 0's attention input is exactly
$x_i = \\mathrm{{rms}}_1(W_E[t_i])$ (pre-norm, first block), so per-head attention is recomputed
directly, including the sink slot (GPT-OSS convention): with per-head logits
$\\ell_{{ij}} = q_i\\cdot k_j/\\sqrt{{128}}$ and the learned scalar $s_h$,

$$A_{{ij}} = \\frac{{e^{{\\ell_{{ij}}}}}}{{e^{{s_h}} + \\sum_{{j'\\le i}} e^{{\\ell_{{ij'}}}}}},\\qquad
A_{{i,\\mathrm{{sink}}}} = \\frac{{e^{{s_h}}}}{{e^{{s_h}} + \\sum_{{j'\\le i}} e^{{\\ell_{{ij'}}}}}} .$$

The sink column is dropped before the value average, so sink mass = attention paid to zero.
Compute: `hide/compute.py` → `hide/cache/l0_attn_stats.npz` and `hide/ov_compute.py` →
`hide/cache/ov_stats.npz`; figures/tables: `hide/report.py`.
All positions kept (EOS included); token-level stats use query positions ≥ 64 and, for
received mass, keys with a full 64-query following window.

## Head summary

| head | sink logit $s_h$ | mean sink mass (qpos ≥ 64) | mean content mass | mean $n_\\mathrm{{eff}}$ | content share $d\\le 8$ | $d\\le 32$ |
|---:|---:|---:|---:|---:|---:|---:|
{head_rows}

$n_\\mathrm{{eff}} = e^H$ of the full (keys + sink) distribution; content share = fraction of
non-sink mass within query−key distance $d$.

**The six heads split into three groups:**

- **Heads 0, 2, 5 — near-pure previous-token heads, parked in the sink ~90% of the time.**
  Their content mass is dominated by $d = 1$ (offset profile below), with 81–86% of it within
  $d \\le 8$; $n_\\mathrm{{eff}} \\approx 1.7$ means a typical query effectively touches the sink
  plus at most one key. They act as *conditional bigram heads*: for most tokens they write
  ~nothing (sink), and wake up on specific trigger tokens (tables below).
- **Head 1 — the context head.** Only 0.09 mean sink mass, $n_\\mathrm{{eff}} \\approx 58$, and
  the flattest distance profile (only 62% of content within $d \\le 32$, measurable mass at
  $d > 256$). This is the one L0 head that reads broadly by default.
- **Heads 3, 4 — intermediate.** Sink mass 0.56 / 0.75, $n_\\mathrm{{eff}}$ 18 / 6.8, medium-range
  profiles (58% / 71% of content within $d \\le 32$).

## Spread of sink mass across individual queries

![per-query sink mass histograms](l0_sink_mass_query_hist.png)

![per-query sink mass histograms, linear y](l0_sink_mass_query_hist_linear.png)

Same data twice: log y (top; shows the tails) and linear y (bottom, per-head y scale;
shows where the bulk actually sits). Distribution of $A_{{i,\\mathrm{{sink}}}}$ over all ~{sink_hist[0].sum() / 1e3:.0f}k individual
queries (positions ≥ 64; log y, bin width 0.01). Tail fractions:

| head | P(sink < 0.1) | P(sink < 0.5) | P(sink > 0.9) | P(sink > 0.98) |
|---:|---:|---:|---:|---:|
{qhist_rows}

The means hide very different shapes:

- **Heads 0, 2, 5**: the bulk sits in a sharp peak at 0.93–0.99 with a hard upper edge just
  below 1 (the $d{{=}}1$ key is always present, so a little always leaks to the previous
  token), and the low-sink side is a long, roughly exponential tail — ~1% of queries below
  0.5, essentially none below 0.1. The tail is the conditional-bigram behavior: the rare
  woken-up queries (the trigger tokens of the tables below) pull most of their mass out of
  the sink, but there is no separate low-sink mode.
- **Head 1** is the mirror image: peak at 0.02–0.05, monotone decay, nothing above ~0.75
  (P(sink > 0.5) = 0.3%) at these positions — the high-sink behavior it shows in the position
  plot lives entirely in the early-context transient, which is excluded here.
- **Head 3** is genuinely **broad** — closest to flat of all heads: a wide hump peaking near
  0.6 with substantial probability everywhere between ~0.1 and ~0.85, and almost nothing
  above 0.9. Its 0.56 mean is a real mixture of query-by-query engagement levels, not a
  fixed operating point.
- **Head 4** sits between the two regimes: a main lobe at 0.8–0.95 (28% of queries above
  0.9) plus a fat flat tail across the whole range and a small shoulder near 0.15 — 15% of
  its queries are below 0.5.

## Is it position dependent?

![sink mass vs position](l0_sink_mass_vs_position.png)

Mostly **no** — with two systematic exceptions:

1. **Very early positions.** At $i = 0$ the only choices are self and sink; all heads put
   0.68–0.99 on the sink there (head 1: 0.68 → it self-attends with the rest). Head 1's sink
   mass then decays quickly ({sink_pos[1, 1]:.2f} at $i{{=}}1$, {sink_pos[1, 8]:.2f} at $i{{=}}8$,
   {sink_pos[1, 32]:.2f} at $i{{=}}32$) as real context accumulates: the sink acts exactly as
   designed, absorbing mass while there is nothing to attend to yet.
2. **A slow drift in heads 3 and 4.** Head 3 falls from {sink_pos[3, 64:128].mean():.2f}
   (positions 64–128) to {sink_pos[3, 384:].mean():.2f} (384–511), head 4 from
   {sink_pos[4, 64:128].mean():.2f} to {sink_pos[4, 384:].mean():.2f} — longer context gives
   these medium/long-range heads more worth attending to, so they leave the sink slightly
   more often. Heads 0, 2, 5 are flat to < 0.01 over the same span.

There is **no residual position-0 attention sink**: mass on key 0 from queries ≥ 64 is
≤ 0.0008 in every head. The built-in sink slot fully replaces the emergent
first-token sink of the original pile_4l model (which parked 20–30% per head on key 0 in
its layers 2–3, per `endoftext-pos0/`).

## Distance profile

![offset profile](l0_offset_profile.png)

Mean mass at distance $d$ (conditional on the query having a key at that distance). All heads
peak at **$d = 1$** — layer 0 is previous-token-flavored across the board — but the decay
rates differ enormously: heads 0/2/5 fall ~2 orders of magnitude by $d \\approx 10$, head 1
decays roughly like a power law and still has ~2×10⁻⁴ mean mass per key at $d \\approx 448$
(times ~hundreds of far keys — that's where its $n_\\mathrm{{eff}} \\approx 58$ lives).
Self-attention ($d = 0$, markers) is weak in all heads; head 1 is the only one where it is
comparable to $d = 1$. The uptick at $d \\gtrsim 300$ in heads 3/4 is far queries putting
mass on early-chunk keys (positions ≲ 60) — a mild residual chunk-start preference in the
*content* attention of the two medium-range heads, even though key 0 itself receives ≈ 0.

![logit offset profile](l0_offset_logits.png)

The same profile in **pre-softmax logit** units, $\\langle \\ell_{{i,i-d}}\\rangle$ with
$\\ell_{{ij}} = q_i\\cdot k_j/\\sqrt{{128}}$ (linear y — logits can be negative), with each
head's sink logit $s_h$ dotted in its color. This is the comparison the softmax actually
makes: a single key at distance $d$ beats the sink where its logit exceeds the dotted line
($A_{{ij}}/A_{{i,\\mathrm{{sink}}}} = e^{{\\ell_{{ij}} - s_h}}$). The striking fact: **no
head's mean logit reaches its sink logit at any distance** — even head 1's $d{{=}}1$ mean
({offset_logit[1, 1]:.2f}) sits a full logit below its $s_1 = {sink_logits[1]:.2f}$. The heads
escape the sink in two different ways: **head 1 by aggregation** — its logits decay so slowly
(still ≈ 0 at $d \\approx 10$, only ≈ −5 at $d \\approx 500$) that hundreds of keys sum to far
more than the single sink term (500 keys at $e^0$ vs $e^{{2.13}} \\approx 8.4$) — and **heads
0/2/5 by fluctuations**: their mean logits plunge to −10…−15 at range (far keys are actively
suppressed, guaranteeing the sink wins by default) and even their $d{{=}}1$ means are −1…−2.5,
so all their content attention comes from rare specific query/key token pairs whose logits
spike far above $s_h$ — the exponential low-sink tail of the query histograms above. Head 4
has both the largest sink logit (3.26) and the *lowest* $d{{=}}1$ mean logit — its
previous-token peak is the weakest, matching its more distributed profile.

## Which query tokens attend to content vs mostly sink?

![token sink histograms](l0_token_sink_hist.png)

Per-query-token mean sink mass over the {len(sel)} token ids with ≥ {MIN} occurrences at
positions ≥ 64. The split is strikingly **syntax vs prose**:

**Lowest sink mass (these tokens' queries actually search the context) — all-head mean:**

{sink_table(low_ids)}

**Highest sink mass (these tokens ask layer 0 for nothing):**

{sink_table(high_ids)}

- The content-attending queries are **code / math / markup structure tokens**: closing and
  opening delimiters (`'>'`, `'<'`, `'}}'`, `'{{'`, `');'`, `'),'`), attribute/equals signs,
  quotes, backslashes, indentation runs, tabs and newlines. These are exactly the tokens whose
  local meaning depends on matching structure earlier in the line/expression. Head-specific
  versions of the same story: head 3's strongest content queries are tab ({ms[3, np.where(sel == 186)[0][0]]:.2f} sink)
  and `'<'`, head 4's are `' to'` ({ms[4, np.where(sel == 281)[0][0]]:.2f}) and newline
  ({ms[4, np.where(sel == 187)[0][0]]:.2f}), head 5's are `'-'`, `' of'`, `':'` and indentation.
- The sink-heavy queries are **plain-prose tokens**: sentence-initial capitals and openers
  (`' In'`, `' The'`, `' It'`, `' This'`), single capital letters (initials), and common
  adverbs/particles/quantifiers (`' just'`, `' all'`, `' out'`, `' up'`, `' first'`,
  `' more'`, `' which'`). For ordinary running text, layer-0 attention (beyond the
  previous-token peak) has little to offer, and the sink absorbs the mass.
- `'<|endoftext|>'` is the single most sink-heavy query (mean {agg[np.where(sel == 0)[0][0]]:.2f}) —
  at a document start the preceding context is by definition irrelevant, and the built-in sink
  lets every head express that directly (in pile_4l this required the emergent massive-vector
  machinery).

## Which tokens receive a lot of attention?

Received mass = for key $j$, the mean of $A_{{j+d,\\,j}}$ over the following {window} queries
($d = 1..{window}$; self excluded, key position ≥ 1, full window required). Top 20 by all-head
mean:

{recv_table(recv_ids)}

- **`'<|endoftext|>'` is by far the strongest attention magnet** — head 1 gives a preceding
  EOS {mr[1, np.where(selk == 0)[0][0]]:.3f} of each following query's mass on average
  (≈ 14× a uniform share of its window), heads 3/4 {mr[3, np.where(selk == 0)[0][0]]:.3f}/{mr[4, np.where(selk == 0)[0][0]]:.3f}.
  Even with a built-in sink slot available, the model still marks document boundaries with
  real attention: EOS-as-key is *read* (boundary information), not just used as a parking spot.
  (As a *query*, EOS sinks: {eos_sink} per head — consistent with `eos-isolation`'s finding
  that L0 reads boundaries while later layers enforce isolation.)
- The other top receivers are again **structure openers**: `'//'`, `' [@'`, `' {{'`, `'<'`,
  `'){{'`, `'if'`, `'_{{'`, `'mathcal'`, `'$'`, `'**'`, `'\\n\\n'`, opening quotes — the left
  ends of constructions whose right ends (the content-attending queries above) need them.
  Layer 0 looks like a **local bracket/construction-matching + previous-token layer**, with head 1
  additionally supplying broad context.
- The weakest receivers are mid-word fragments (`'o'`, `'ing'`, `'s'`, `'a'`) — nothing looks
  back at pieces of words.

## Which tokens — head by head

Per-head versions of the token tables above, heads ordered by **ascending mean sink mass**
(most content-attending first). Same conventions: query-token sink mass over positions ≥ 64
(≥ {MIN} occurrences), received mass = mean mass from the following {window} queries. Each
head also gets its own distance profile (log–log full range + linear zoom on $d \\le 32$),
in attention-mass and in logit units. In the zoom panels the error bars are per-query
quantile bands (thick = p25–p75, thin = p5–p95) — note the distributions are heavily
skewed, so the mean line can sit *outside* the interquartile bar (e.g. head 4's mass at
$d{{=}}1$: the mean is dominated by the strong-attention tail while the median query gets
far less), which is exactly the mean-mass-vs-mean-logit discrepancy discussed above.

{perhead_md}

## Do the sink-heavy heads compensate with a large OV circuit?

**Yes — and it is still true that they do little most of the time; both halves of the
question are right.** Per head, define the actual residual write of query $i$ (the sink slot
contributes exactly zero value):

$$w_i^h = W_O^h \\sum_{{j\\le i}} A_{{ij}}\\, v_j,\\qquad
\\lVert w_i^h\\rVert \\;\\lesssim\\; c_i \\cdot \\max_j \\lVert W_O^h v_j\\rVert,
\\qquad c_i = 1 - A_{{i,\\mathrm{{sink}}}} .$$

| head | top OV $\\sigma_1$ | $\\lVert W_O^h W_V^h\\rVert_F$ | mean $\\lVert W_O^h v_j\\rVert$ | mean $c_i$ | median $\\lVert w\\rVert$ | mean $\\lVert w\\rVert$ | p99 $\\lVert w\\rVert$ | mean $\\lVert w\\rVert$ at $c\\approx 0.5$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{ov_rows}

(Reference scale: the residual stream at this point is the raw embedding, mean norm
{float(OV['emb_norm_mean']):.2f}.)

![OV write norms](l0_ov_write_norms.png)

- **The compensation is real and large.** The sink-heavy heads 0/2/5 have the *biggest* OV
  circuits in the layer: per-key write norm $\\lVert W_O^h v_j\\rVert$ averages 2.6–3.1 vs
  **0.81 for head 1** — 3–4× larger — and the same factor shows in the static
  $\\lVert W_O W_V\\rVert_F$ (3.3–3.6 vs 0.58). The right panel makes it graphic: per unit of
  content mass, heads 0/2/5 write ~2 units of residual norm, head 1 only ~0.6 (its writes are
  further shrunk by averaging over ~58 keys, which partially cancel). At the same content
  mass $c = 0.5$, heads 0/2/5 write ≈ 1.0 — bigger than head 1 *ever* writes (its p99 is
  0.7, because its softmax never concentrates).
- **And yet, most of the time they still do relatively little.** Median write norms: heads
  0/2/5 = 0.08–0.13 vs head 1 = 0.50 — the 3–4× OV boost only partially offsets the ~10×
  smaller typical content mass. Relative to the residual stream (norm ≈ 0.77) the sinky
  heads' typical write is ~10–16%, i.e. a genuine near-no-op, while head 1's typical write
  is ~65% of the stream norm.
- The two facts together are exactly the **conditional-head design**: a large OV multiplier
  means the head does not need much softmax mass to act — waking up to $c \\approx 0.25$
  already writes ~0.5, comparable to head 1's typical output — and the sink keeps the head
  silent (write ∝ $c$) the rest of the time. The write norm is almost perfectly linear in
  content mass (right panel), so for these heads the sink fraction *is* the head's activation
  level, with heavy tails: their p99 writes (1.0–1.1) exceed anything head 1 produces.
- Heads 3/4 are again intermediate, and head 3 is notably **weak per unit mass** (~0.3–0.5
  per unit $c$, the flattest line): its broad medium-range attention averages many keys, so
  even at high content mass its net write stays small — closer to head 1's
  averaging-and-cancelling regime than to the sharp bigram heads.

## Take-aways

1. Layer 0 of the sink model spends most of its attention budget on the built-in sink: 4 of 6
   heads park 75–91% of their mass there (weighted by the learned logits $s_h$ = {", ".join(f"{s:.2f}" for s in sink_logits)}).
2. The sink usage is essentially **token-driven, not position-driven**: profiles are flat in
   query position (except the trivial early-context transient and a mild drift in heads 3/4),
   and per-token sink mass separates syntax tokens (attend) from prose tokens (sink).
3. The emergent pos-0 sink of pile_4l is gone (key-0 mass ≈ 0), but **EOS keys still attract
   outsize genuine attention** — boundary reading survives; boundary *parking* moved into the
   architecture.
4. Head roles: 0/2/5 = sink-defaulted previous-token/bigram heads; 1 = broad context head
   (the only head that is mostly *not* in the sink); 3/4 = medium-range, mildly
   position-sensitive.
5. The sink-heavy heads carry the **largest OV circuits** (per-key writes 3–4× head 1's):
   small post-softmax content mass is partially compensated by a large value/output
   multiplier, making them low-duty-cycle but strong-when-active conditional heads — though
   their *typical* write is still a near-no-op (~10–15% of the stream norm vs ~65% for
   head 1).
"""

(OUT / "report.md").write_text(report, encoding="utf-8")
print("wrote report.md + 3 figures")
