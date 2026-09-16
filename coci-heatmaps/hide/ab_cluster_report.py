"""Render AB-cluster-report.md (one level up): what the large alive co-CI>0.9
clusters of the two new decompositions fire on — pos-0 sink vs EOS vs
token-based classification tables + member-id appendix.

Inputs: cache/big_clusters.json (big_clusters_extract.py) and
cache/big_clusters_{newA,newB}.npz (big_clusters_modal.py).

Usage: python coci-heatmaps/hide/ab_cluster_report.py   (~1 min)
"""

import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from load import load_tokenizer  # noqa: E402
from report import MATS  # noqa: E402

EOS_ID = 0
COEFF = {"newA": "6.6e-5", "newB": "6.6e-6"}
RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}

INTRO = """\
# newA/newB large co-CI clusters — what do they fire on?

Classification of every **alive co-CI>0.9 cluster with >= 5 members** in the two
new 800k-step decompositions (newA = `p-8383f5e5`, minimality coeff 6.6e-5;
newB = `p-4d9a6a12`, coeff 6.6e-6; clusters as in
`newA/report_alive_clustered09.md` / `newB/...`, member ids re-derived by the
same threshold rules in `hide/big_clusters_extract.py`).

For every member component: per-position CI>0.1 fire counts and the tokens
fired on, over the 4,000 cached Pile rows (2.05M tokens; Modal job
`hide/big_clusters_modal.py` → `hide/cache/big_clusters_{newA,newB}.npz`).
Classes (share of the cluster's total member-fires):

- **pos-0** — > 50% of fires at position 0 (chunk start). These are
  position-0 attention-sink machinery: they fire once per row at the first
  position *whatever token is there* — their fired-token histogram is just the
  corpus unigram distribution at position 0, so no top-token column is shown.
- **EOS** — > 50% of fires on mid-sequence `<|endoftext|>` tokens.
- **early-pos** — > 50% of fires at positions 0–7 but not position 0 alone.
- **token** — everything else; the top fired tokens are shown.

Columns: n = members; mean CI = sample mean over members (the once-per-row
signature is ~4000/2.05M ≈ 2.0e-3); fires = total member-fires (CI > 0.1);
pos0/EOS/p1–7 = share of fires at position 0 / on mid-seq EOS / at positions
1–7; f/row = fires per member per row (1.00 = exactly once per row).
Rows are grouped by site (layer, then q/k/v/o/c_fc/down_proj; thick rule
between sites), largest cluster first within a site.
"""


def tok_fmt(tokenizer, th, k=6):
    th = th.copy()
    th[EOS_ID] = 0
    top = np.argsort(-th)[:k]
    parts = []
    for t in top:
        if th[t] == 0:
            break
        parts.append(f"{html.escape(repr(tokenizer.decode([int(t)])))}"
                     f"×{int(th[t])}")
    return ", ".join(parts)


def classify_rows(d, name_clusters, tokenizer):
    """(site, ids, label, mean_ci, fires, pos0/EOS/p1-7 shares, f/row, tokens)
    per cluster, site-sorted (layer, matrix order, size desc)."""
    sites: dict[str, list] = {}
    for cl in name_clusters:
        sites.setdefault(cl["site"], []).append(cl["ids"])
    rows_out = []
    for s, cls in sites.items():
        pc = d[f"{s}|pos_counts"]
        eos = d[f"{s}|eos"]
        off = 0
        for k, c in enumerate(cls):
            n = len(c)
            sl = slice(off, off + n)
            off += n
            th = d[f"{s}#{k}|tok"]
            total = int(pc[sl].sum())
            f0 = pc[sl, 0].sum() / total
            fe = eos[sl].sum() / total
            f17 = pc[sl, 1:8].sum() / total
            label = ("pos-0" if f0 > 0.5 else "EOS" if fe > 0.5 else
                     "early-pos" if f0 + f17 > 0.5 else "token")
            mean_ci = [cl["mean_ci"] for cl in name_clusters
                       if cl["site"] == s and cl["ids"] == c][0]
            tops = "—" if label == "pos-0" else tok_fmt(tokenizer, th)
            rows_out.append((s, c, label, mean_ci, total, f0, fe, f17,
                             total / n / 4000, tops))

    def site_key(s):
        _, l, mat = s.split(".", 2)
        return int(l), MATS.index(mat)

    rows_out.sort(key=lambda r: (site_key(r[0]), -len(r[1])))
    return rows_out


def html_table(rows_out):
    """Site-grouped table: thick rule between sites, thin between clusters."""
    header = ["site", "n", "class", "mean CI", "fires", "pos0", "EOS",
              "p1–7", "f/row", "top fired tokens (excl. EOS)"]
    right = {1, 3, 4, 5, 6, 7, 8}  # numeric columns
    tbl = ['<table style="border-collapse:collapse">',
           "<tr>" + "".join(
               f'<th style="padding:2px 8px;border-bottom:2px solid #555;'
               f'text-align:{"right" if i in right else "left"}">{h}</th>'
               for i, h in enumerate(header)) + "</tr>"]
    prev_site = None
    for s, c, label, mci, total, f0, fe, f17, fpr, tops in rows_out:
        border = ("3px solid #555" if s != prev_site else "1px solid #ccc")
        prev_site = s
        cells = [s, str(len(c)),
                 label if label == "pos-0" else f"<b>{label}</b>",
                 f"{mci:.1e}", str(total), f"{f0*100:.1f}%",
                 f"{fe*100:.1f}%", f"{f17*100:.1f}%", f"{fpr:.2f}", tops]
        tbl.append("<tr>" + "".join(
            f'<td style="padding:2px 8px;border-top:{border};'
            f'text-align:{"right" if i in right else "left"}">{v}</td>'
            for i, v in enumerate(cells)) + "</tr>")
    tbl.append("</table>")
    return "\n".join(tbl)


def appendix_lines(name, rows_out):
    lines = [f"### {name}", ""]
    for s, c, label, *_ in rows_out:
        lines.append(f"- **{s}** ({len(c)}, {label}): "
                     + ", ".join(str(i) for i in c))
    return lines + [""]


def main() -> None:
    tokenizer = load_tokenizer("pile_4l")
    clusters = json.loads((HERE / "cache" / "big_clusters.json").read_text())
    lines = [INTRO]
    appendix = ["## Appendix: cluster members", ""]

    for name in ("newA", "newB"):
        d = np.load(HERE / "cache" / f"big_clusters_{name}.npz")
        rows_out = classify_rows(d, clusters[name], tokenizer)
        n_pos0 = sum(1 for r in rows_out if r[2] == "pos-0")
        lines += [
            f"## {name} (`{RUNS[name]}`, coeff {COEFF[name]}) — "
            f"{len(rows_out)} clusters, {n_pos0} pos-0",
            "",
            html_table(rows_out),
            "",
        ]
        appendix += appendix_lines(name, rows_out)

    lines += FINDINGS + appendix
    out = HERE.parent / "AB-cluster-report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


FINDINGS = ["""\
## Findings

**Size sorts the classes.** Every cluster with ≥ 19 members is pos-0 sink
machinery, except newB's 18-comp spam-punctuation cluster (below). All pos-0
clusters share the once-per-row signature (mean CI ≈ 2.0e-3, f/row ≈ 1) and
appear in every layer and every matrix type — with the weak minimality
pressure, the whole pos-0 sink circuit (keys, values, o_proj, MLP trigger and
write stages) duplicates into big co-CI blocks. Sub-flavors: newB has three
small *dual-site* sink clusters firing ~1.4–1.6×/row with a 22–26% mid-seq-EOS
share (h.3.attn.k_proj 7, h.3.attn.o_proj 7, h.2.attn.k_proj 6 — pos-0 AND EOS,
like the old decomposition's dual subgroups), and the big h.2/h.3 v_proj
clusters carry milder 6–12% EOS side shares. A few low-CI L0 pos-0 clusters
(f/row 0.02–0.36) fire at pos 0 only on a token-conditioned subset of rows.

**EOS clusters** (newA 4, newB 6) all sit at f/row ≈ 0.35 = the mid-seq EOS
density of the sample (1,412 EOS in 4,000 rows). Their few non-EOS fires are
document-boundary-flavored tokens: form feeds '\\x0c', '\\n\\n\\n' runs, '---',
'######'. The 15/13-comp h.1.attn.v_proj clusters are the analogues of the old
decomposition's 68-comp h.1.attn.v_proj EOS block.

**Token clusters** (newA 13, newB 14) have crisp, mostly-L0/L1 identities at
low fire rates:

- **':'** — several 5–11-comp clusters in both decomps (h.0.mlp.c_fc/down_proj,
  h.1.attn.v_proj).
- **Horizontal rules** '------' / '~~~' / '======' (markdown/RST section
  underlines) — three matrices in newA, one in newB.
- **'...' ellipsis** (newA h.2.attn.k_proj 5), **'Q'** (the top document-opener
  token; newA/newB h.0.attn.k_proj), **' prob' + 'ubotu'/'ubottu'** (newA
  h.0.attn.k_proj 6 — Ubuntu IRC-log machinery), **'Copyright'** (newA
  h.0.mlp.down_proj 5), **' replacement'** (newA h.0.attn.v_proj 5),
  **' from'** (newB h.0.attn.o_proj 5 + h.1.attn.k_proj 5), **'Table'/'Figure'**
  (newA h.1.attn.k_proj 5), **'Category'/'External'** (newB h.0.attn.v_proj 6 —
  Wikipedia markup), **' !'** (newB h.0.mlp.down_proj 6), small **'\\n'**
  clusters.
- newB has two *dense* L0 c_fc token clusters: **' of'** (5 comps,
  mean CI 1.4e-2, 7.9 fires/row/comp — essentially every ' of'/'of'/' between'
  in the corpus) and **newline + indentation** (5 comps, 2.9/row: '\\n', '  ',
  ' *', '//' — code/markup line starts).
- **newB h.2.mlp.c_fc 18-comp cluster = sentence-final punctuation in
  spam/word-salad documents**: '.' 86% of fires (rest ';?:!\\n'), but only 109
  distinct sites concentrated in 18 of the 4,000 rows — all SEO-spam / garbled
  machine-generated English, usually all 18 members firing together (fire sites:
  `hide/cache/punct_cluster_newB.npz` via `hide/punct_cluster_modal.py`).

**Cross-matrix duplicates.** Several token clusters recur in *different
matrices* with identical fire counts (newB ':' ×1210 in h.0.mlp.c_fc,
h.0.mlp.down_proj and h.1.attn.v_proj; newA '------'/'~~~'/'======' in
h.0.mlp.c_fc, h.0.mlp.down_proj and h.1.attn.v_proj; ' from' ×265 in newB
h.0.attn.o_proj and h.1.attn.k_proj) — the same trigger sites detected at
successive stages of one circuit, so the co-CI blocks are per-matrix slices of
cross-matrix mechanisms.
"""]

if __name__ == "__main__":
    main()
