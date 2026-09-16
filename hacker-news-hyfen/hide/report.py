"""Render hacker-news-hyfen/report.md + report.html from cache/hn_ci.npz
(Modal output), cache/act_cand.npz and cache/hn_positions.npz.

Selection: a component is in the table iff its mean CI over the 12
hacker-news hyphens is > 0.1.  Sorted by matrix (layer order, then
q/k/v/o/c_fc/down_proj), then by that mean, descending.

The component table is an inline-styled HTML heatmap table (gradient cell
shading, thick rules between matrices), embedded in report.md and duplicated
as the standalone report.html.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_tokenizer  # noqa: E402

MATRICES = [f"h.{l}.{m}" for l in range(4)
            for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                      "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]
THRESH = 0.1

ci = np.load(HERE / "cache/hn_ci.npz")
pos = np.load(HERE / "cache/hn_positions.npz")
cand_rc, hn_rc, n_hyph = pos["cand_rc"], pos["hn_rc"], int(ci["n_hyph"])
hn_cols = np.array([k for k, rc in enumerate(cand_rc)
                    if any((rc == h).all() for h in hn_rc)])
assert len(hn_cols) == len(hn_rc) == 12 and n_hyph == len(pos["hyphen_rc"])

# per-matrix stats -> selected rows
act = np.load(HERE / "cache/act_cand.npz")
assert (act["cand_rc"] == cand_rc).all()
rows_out = []
for m in MATRICES:
    ci_hn = ci[f"{m}|CI_cand"][:, hn_cols]            # (C, 12)
    mean_hn = ci_hn.mean(1)
    sel = np.where(mean_hn > THRESH)[0]
    s_all, s_hyph = ci[f"{m}|S_all"], ci[f"{m}|S_hyph"]
    a_hn = act[f"{m}|A"][:, hn_cols].mean(1)
    for c in sel[np.argsort(-mean_hn[sel])]:
        rows_out.append((f"{m}:{c}", mean_hn[c], a_hn[c], s_hyph[c] / n_hyph,
                         ci_hn[c].sum() / s_hyph[c],
                         ci_hn[c].sum() / s_all[c], s_hyph[c] / s_all[c]))

# --- HTML heatmap table for report.html
# Sequential blue ramp (dataviz reference palette, steps 100-700) prepended
# with the light surface as the zero anchor; explicit bg AND text colors on
# every shaded cell so light/dark browser themes both read.
RAMP = ["#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
        "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
        "#104281", "#0d366b"]
RGB = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in RAMP]


def cell(val: float, t: float, fmt: str, sep: str) -> str:
    t = min(max(t, 0.0), 1.0) * (len(RGB) - 1)
    i, f = min(int(t), len(RGB) - 2), t - min(int(t), len(RGB) - 2)
    r, g, b = (round(a + (b_ - a) * f) for a, b_ in zip(RGB[i], RGB[i + 1]))
    ink = "#ffffff" if t / (len(RGB) - 1) > 0.55 else "#0b0b0b"
    return (f'<td style="{sep}background:rgb({r},{g},{b});color:{ink};'
            f'padding:2px 8px;text-align:right">{val:{fmt}}</td>')


a_max = max(r[2] for r in rows_out)
html = ['<table style="border-collapse:collapse">',
        "<thead><tr>"
        + "".join(f'<th style="padding:2px 8px;text-align:{al}">{h}</th>'
                  for h, al in [("Component", "left"),
                                ("mean CI<br>HN hyphens", "right"),
                                ("mean |a|<br>HN hyphens", "right"),
                                ("mean CI<br>all hyphens", "right"),
                                ("tot CI HN /<br>tot CI hyphens", "right"),
                                ("tot CI HN /<br>tot CI anywhere", "right"),
                                ("tot CI hyphens /<br>tot CI anywhere", "right")])
        + "</tr></thead>", "<tbody>"]
prev_m = rows_out[0][0].rsplit(":", 1)[0]
for k, m_hn, a_hn_c, m_hy, r_hh, r_hn, r_hy in rows_out:
    m = k.rsplit(":", 1)[0]
    sep = "border-top:3px solid #52514e;" if m != prev_m else ""
    prev_m = m
    html.append(
        "<tr>"
        f'<td style="{sep}padding:2px 8px"><code>{k}</code></td>'
        + cell(m_hn, m_hn, ".3f", sep)
        + cell(a_hn_c, np.log1p(a_hn_c) / np.log1p(a_max), ".2f", sep)
        + cell(m_hy, m_hy, ".4f", sep)
        + cell(r_hh, r_hh, ".3f", sep)
        + cell(r_hn, r_hn, ".3f", sep)
        + cell(r_hy, r_hy, ".3f", sep)
        + "</tr>")
html += ["</tbody>", "</table>"]

# candidate contexts for the appendix (decoded on the fly); validation column =
# mean CI of the 26 description-derived L0q hn-hyphen group members (L0_q.md)
tok = load_tokenizer("pile_4l")
data_rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
CORE = [1, 20, 21, 33, 56, 64, 83, 104, 120, 146, 154, 158, 170, 184, 228,
        230, 241, 273, 283, 309, 315, 387, 397, 413, 497, 504]
core_mean = ci["h.0.attn.q_proj|CI_cand"][CORE].mean(0)

cand_lines = []
for k, (r, p) in enumerate(cand_rc):
    ids = data_rows[r][:512].tolist()
    ctx = tok.decode(ids[max(0, p - 14):p + 6]).replace("\n", "\\n")
    tag = "HN " if k in hn_cols else "rej"
    cand_lines.append(f"| {tag} | {r} | {p} | {core_mean[k]:.3f} | `{ctx}` |")

lines = [
    "# hacker-news-hyfen — components with CI > 0.1 on hacker-news hyphens",
    "",
    "**Definitions.** A *hyphen* is exactly the token 428 (`' -'`). A"
    " *hacker-news hyphen* is a `' -'` separating title and username in a"
    " hacker-news heading (`Title - username\\n[url]\\n======\\n...`, at"
    " document start). Over the standard 4000 cached Pile rows (4000x512 ="
    f" 2.048M tokens; `context-loss/hide/cache/pile_rows.pt`) there are"
    f" **{n_hyph}** hyphens, of which **{len(hn_rc)}** are hacker-news hyphens"
    " (auto-detected heading candidates minus manual rejects; every candidate"
    " was reviewed by hand — `hide/cache/candidates.txt`).",
    "",
    "**Selection.** All components of all 24 decomposed matrices whose *mean*"
    " CI (lower_leaky) over the 12 hacker-news hyphens exceeds 0.1"
    f" ({len(rows_out)} components). Sorted by matrix, then by that mean,"
    " descending. 'Total CI anywhere' = sum of the component's CI over all"
    " 2.048M positions.",
    "",
    "**Reading notes.**",
    "",
    "- The HN-firing components reach CI ~1 on only 11 of the 12 HN hyphens,"
    " so their mean tops out around 0.85: the heading in row 188 (`...roses."
    "<|endoftext|>Cross platform ... Print Spooler API - yadavrg`) gets"
    " *zero* CI from all of them — it is the only heading where the title"
    " follows `<|endoftext|>` with no newline in between. Row 1333 (title"
    " starting with a stray space) only gets weak CI (~0.14 from the core"
    " group).",
    "- The ratio columns measure specificity: 'tot CI HN / tot CI hyphens'"
    " is the fraction of the component's hyphen CI mass that sits on the HN"
    " ones (how HN-dominated its hyphen firing is); the two '/ tot CI"
    " anywhere' columns: ~1 means the component's"
    " entire CI mass in the sample sits on the (HN) hyphens; ~0 means a"
    " broadly-active component that also lights up here (e.g."
    " h.0.attn.q_proj:28, the dense always-on component, tops the L0q block"
    " by mean but has ratio 0.000). Far more components than the 26"
    " description-derived `hn-hyphen` group members have CI ~1 at HN hyphens"
    " (53 in L0q alone); the ratio columns pick out the HN-*specific* ones.",
    "",
    "**Table.** `mean |a|` = mean activation strength"
    " |a_c| = ‖U_c‖·|V_c·x| over the 12 HN hyphens. Cell shading: blue"
    " gradient over 0→1 for the CI and ratio columns; the |a| column is"
    f" unbounded (max here {a_max:.1f}) and is shaded on a log scale to its"
    " max. Thick rules separate the matrices. (Same table as a standalone"
    " page: [report.html](report.html).)",
    "",
    *html,
    "",
]

lines += [
    "",
    "## The 17 heading candidates",
    "",
    "Auto-detected `Title - username` headings; 'core CI' = mean CI of the 26"
    " description-derived L0q `hn-hyphen` group members (see"
    " `pile-qk-comps/L0/L0_q.md`) at that position. The 5 rejects are cleanly"
    " dead (0.000), confirming the manual calls; row 188 is the"
    " no-newline-after-EOS miss described above.",
    "",
    "| | row | pos | core CI | context (`...title - user`) |",
    "|---|---|---|---|---|",
    *cand_lines,
    "",
    "Scripts: `hide/find_hyphens.py` (position detection) ->"
    " `hide/ci_compute_modal.py` (Modal A10G, CI sums + per-candidate CI for"
    " all 38,912 components) -> `hide/report.py` (this file).",
]

out = ROOT / "hacker-news-hyfen/report.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(rows_out)} components)")

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>hacker-news-hyfen — component table</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px auto; max-width: 900px;
         background: #fcfcfb; color: #0b0b0b; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a19; color: #ffffff; }} }}
  code {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.9em; }}
  th {{ vertical-align: bottom; }}
  p {{ max-width: 72ch; }}
</style></head><body>
<h1>Components with mean CI &gt; 0.1 on hacker-news hyphens</h1>
<p>{len(rows_out)} components of all 24 matrices, over the 12 hacker-news
title|username hyphens in the 4000 cached Pile rows (definitions, selection
criterion and heading list: <a href="report.md">report.md</a>).
Sorted by matrix (thick rules), then by mean CI on HN hyphens, descending.</p>
<p>Cell shading: blue gradient over 0&rarr;1 for the CI and ratio columns;
the |a| column (mean activation strength |a_c| = &Vert;U_c&Vert;&middot;|V_c&middot;x|)
is unbounded (max here {a_max:.1f}) and is shaded on a log scale to its max.</p>
{chr(10).join(html)}
</body></html>
"""
out_html = ROOT / "hacker-news-hyfen/report.html"
out_html.write_text(page, encoding="utf-8")
print(f"wrote {out_html}")
