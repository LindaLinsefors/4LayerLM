"""Render L0-attn-cluster-report.md (one level up): what every >=2-member
alive co-CI>0.9 cluster of the four h.0.attn matrices fires on, for the old
decomposition and newA/newB — same classification and table format as
AB-cluster-report.md (helpers imported from ab_cluster_report.py).

Inputs: cache/l0_attn_clusters.json (l0_attn_clusters_extract.py) and
cache/l0_attn_clusters_{old,newA,newB}.npz (l0_attn_clusters_modal.py).

Usage: python coci-heatmaps/hide/l0_attn_cluster_report.py   (~1 min)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ab_cluster_report import (appendix_lines, classify_rows,  # noqa: E402
                               html_table, load_tokenizer)

NAMES = {"old": "`s-55ea3f9b` (the paper's 400k-step decomposition)",
         "newA": "`p-8383f5e5` (800k steps, minimality coeff 6.6e-5)",
         "newB": "`p-4d9a6a12` (800k steps, minimality coeff 6.6e-6)"}

INTRO = """\
# Layer-0 attention co-CI clusters — what do they fire on?

Classification of **every alive co-CI>0.9 cluster with >= 2 members** in the
four **h.0.attn** matrices (q/k/v/o_proj), for the old decomposition and both
new ones (clusters as in the corresponding report_alive_clustered09.md; member
ids re-derived by the same threshold rules in
`hide/l0_attn_clusters_extract.py`). Same measurements, classes and columns as
[AB-cluster-report.md](AB-cluster-report.md) (per-position CI>0.1 fire counts
and fired-token histograms over the 4,000 cached Pile rows = 2.05M tokens;
Modal job `hide/l0_attn_clusters_modal.py` →
`hide/cache/l0_attn_clusters_{old,newA,newB}.npz`):

- **pos-0** — > 50% of fires at position 0 (chunk start), token-independent;
  no top-token column (it is just the corpus unigram distribution at pos 0).
- **EOS** — > 50% of fires on mid-sequence `<|endoftext|>`.
- **early-pos** — > 50% of fires at positions 0–7 but not position 0 alone.
- **token** — everything else; top fired tokens shown.

Mean CI is the **harvest mean CI** for old (as in its clustered09 report) and
the sample mean over the 2.05M tokens for newA/newB. Rows are grouped by
matrix (q/k/v/o; thick rule between matrices), largest cluster first within a
matrix.
"""


def main() -> None:
    tokenizer = load_tokenizer("pile_4l")
    clusters = json.loads((HERE / "cache" / "l0_attn_clusters.json").read_text())
    lines = [INTRO]
    appendix = ["## Appendix: cluster members", ""]

    for name in ("old", "newA", "newB"):
        d = np.load(HERE / "cache" / f"l0_attn_clusters_{name}.npz")
        rows_out = classify_rows(d, clusters[name], tokenizer)
        counts = {lab: sum(1 for r in rows_out if r[2] == lab)
                  for lab in ("pos-0", "EOS", "early-pos", "token")}
        lines += [
            f"## {name} = {NAMES[name]} — {len(rows_out)} clusters "
            f"({', '.join(f'{v} {k}' for k, v in counts.items() if v)})",
            "",
            html_table(rows_out),
            "",
        ]
        appendix += appendix_lines(name, rows_out)

    lines += FINDINGS + appendix
    out = HERE.parent / "L0-attn-cluster-report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


FINDINGS = ["""\
## Findings

**At >= 2 members the class mix flips.** Where the >= 15-member clusters of
AB-cluster-report.md were almost all pos-0, the small L0-attention clusters
are dominated by token classes: old 51 token / 37 pos-0 / 5 EOS / 2 early-pos,
newA 38 / 22 / 3 / 0, newB 41 / 12 / 3 / 0. The pos-0 machinery concentrates
in v/o_proj: in old it is fragmented (27 of the 39 o_proj clusters are pos-0,
all small — matching pos0_components.md's finding that 260/449 fired L0 o_proj
comps are early-locked but diffuse), while newA/newB consolidate it into a few
big blocks (newB: one 23-comp o cluster, one 17-comp v cluster). EOS clusters
appear in all three decompositions, always at f/row ≈ 0.35 = the sample's EOS
density.

**Known old-decomposition mechanisms are recovered exactly.** The 26-comp old
q cluster is the hacker-news-hyphen block (fires ' -' 284 ≈ 26 members × 11 of
the 12 HN hyphens; see hacker-news-hyfen/); the 4-comp early-pos q cluster on
' the'/' The' is the chunk-start-'the' four-way-split mechanism (q:38/212/270/
345 from pile-qk-comps); the 'Q' machinery (q 3, k 16, v 9 comps) is the
StackExchange-Q genre block; the dense old k pair at 0.94 fires/row on ' de',
' que', 'ä', ' los', ' het' is a non-English-text detector.

**Token clusters reproduce across decompositions.** A dozen identities recur
in all three independently trained decompositions (old 400k Torch vs newA/newB
800k JAX): 'Q', the ubotu-IRC pair (' prob' + 'ubotu'/'ubottu'),
' replacement', ' Which', '://' URLs, LaTeX 'frac' / 'label' / 'begin', wiki
'Category', markdown/RST horizontal rules ('------'/'~~~'/'======'), ':'
clusters, ' from', form-feed '\\x0c' page breaks, and the EOS blocks — strong
evidence these are real mechanisms of the target model, not decomposition
artifacts.

**Genre/collocation clusters** (mostly new-decomp): newB k 'Let'/'Suppose'/
'Solve' (math problem statements) and v ' smallest'/' biggest'/' nearest'/
' closest'/'Sort'/'Put' (math-exercise prompts); newB o ' CD'/'+'/'4'/'8'
(CD4+/CD8+ immunology); newA q ' supported'/' funded' (acknowledgments
sections), ' family'/' population'/' village' (census demographics), v
' relates' (patent boilerplate "the present invention relates to..."); newA/
old v clusters on Hungarian ('ő', ' hogy', ' és') and Lithuanian ('ė', 'ų')
characters; '</' HTML closing tags; newB k 'select'/'SELECT' (SQL).
"""]

if __name__ == "__main__":
    main()
