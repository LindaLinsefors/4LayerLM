"""Render C-pos-0/report.md: decomposition C's pos-0 components.

Selection (from sink-models/hide/cache/pos_fires_sink_C.npz): alive (sample
mean CI > 1e-6; standing rule — dead components are always excluded) AND total
fires (CI > 0.1 over the 4,000 cached Pile rows) >= 20 AND > 30% of fires at
position 0 (counts tables shown for both the > 90% and > 30% cuts).  The fires
cut alone already implies mean CI > ~1e-6, so the alive filter is explicit
belt-and-suspenders.  Per component: top firing tokens (share of summed CI per input
token id, from cache/top_tokens_Cpos0.npz — Modal job top_tokens_modal.py)
and percent of total CI mass at positions 0..10 (from the Sp sums), shown as
color-shaded HTML tables (inline styles, render-anywhere).

Run:  python C-pos-0/hide/report.py   (default Python 3.11, needs the two caches)
"""

import html as html_mod
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from load import load_tokenizer  # noqa: E402

MIN_FIRES = 20
SHARE = 0.3        # per-matrix tables list this cut
SHARE_STRICT = 0.9  # counts shown for both cuts
ALIVE = 1e-6       # sample mean CI above this = alive; dead always excluded
MTYPES = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
          "mlp.c_fc", "mlp.down_proj"]
NPOS = 11  # pos 0..10 columns

# Sequential blue ramp (dataviz reference palette) with the light surface as
# zero anchor; explicit bg AND ink on every shaded cell so light/dark themes
# both read (same style as hacker-news-hyfen).
RAMP = ["#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
        "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
        "#104281", "#0d366b"]
RGB = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in RAMP]


def cell(text: str, t: float, sep: str = "") -> str:
    t = min(max(t, 0.0), 1.0) * (len(RGB) - 1)
    i = min(int(t), len(RGB) - 2)
    f = t - i
    r, g, b = (round(a + (b_ - a) * f) for a, b_ in zip(RGB[i], RGB[i + 1]))
    ink = "#ffffff" if t / (len(RGB) - 1) > 0.55 else "#0b0b0b"
    return (f'<td style="{sep}background:rgb({r},{g},{b});color:{ink};'
            f'padding:2px 6px;text-align:right;white-space:nowrap">{text}</td>')


def plain(text: str, align: str = "left") -> str:
    return (f'<td style="padding:2px 8px;text-align:{align};'
            f'white-space:nowrap">{text}</td>')


def main() -> None:
    zp = np.load(ROOT / "sink-models" / "hide" / "cache" / "pos_fires_sink_C.npz")
    zt = np.load(HERE / "cache" / "top_tokens_Cpos0.npz")
    tokzr = load_tokenizer("pile_4l")
    t_sites = [str(s) for s in zt["sites"]]
    t_cols = {(t_sites[s], int(i)): k
              for k, (s, i) in enumerate(zip(zt["site"], zt["ids"]))}
    ci_tok = zt["ci"]  # (vocab, K) summed CI per input token id

    mods = [f"h.{l}.{m}" for l in range(4) for m in MTYPES]
    n_tokens = 4000 * 512

    def select(cut: float) -> dict:
        sel = {}
        for m in mods:
            fp = zp[f"{m}|Fp"]
            tot = fp.sum(1)
            share = np.divide(fp[:, 0], tot, out=np.zeros(len(fp)),
                              where=tot > 0)
            alive = zp[f"{m}|Sp"].sum(1) / n_tokens > ALIVE
            sel[m] = np.nonzero(alive & (tot >= MIN_FIRES) & (share > cut))[0]
        return sel

    selected = select(SHARE)
    strict = select(SHARE_STRICT)
    n_total = sum(len(v) for v in selected.values())
    assert n_total == len(zt["ids"]), (n_total, len(zt["ids"]))

    out = []
    out.append("# Decomposition C: position-0 components\n")
    out.append(
        f"**Decomposition C** = `p-d60af588` (attention-sink target `t-87f91319`, "
        f"seed 0).  A component's **fires** are the tokens where its CI > 0.1, "
        f"over the 4,000 cached Pile rows (2.05M tokens; chunk boundaries cut "
        f"documents at random points, so the position-0 token distribution ≈ the "
        f"corpus distribution).  Selected here: every **alive** component "
        f"(sample mean CI > 1e-6; dead components are excluded, as always) with "
        f"≥ {MIN_FIRES} fires and **> {SHARE:.0%} of its fires at position 0** — "
        f"{n_total} components.  Data: `sink-models/hide/cache/pos_fires_sink_C.npz` + "
        f"`hide/cache/top_tokens_Cpos0.npz`; rendered by `hide/report.py`.\n")

    def counts_table(sel: dict, caption: str) -> None:
        out.append(f"### {caption}\n")
        out.append("| | " + " | ".join(f"`{m}`" for m in MTYPES) + " | total |")
        out.append("|" + "---|" * (len(MTYPES) + 2))
        for l in range(4):
            row = [str(len(sel[f"h.{l}.{m}"])) for m in MTYPES]
            out.append(f"| **L{l}** | " + " | ".join(row)
                       + f" | **{sum(int(x) for x in row)}** |")
        col_tot = [sum(len(sel[f"h.{l}.{m}"]) for l in range(4)) for m in MTYPES]
        out.append("| **total** | " + " | ".join(f"**{c}**" for c in col_tot)
                   + f" | **{sum(col_tot)}** |")
        out.append("")

    out.append("## Counts per matrix\n")
    counts_table(strict, f"> {SHARE_STRICT:.0%} of fires at position 0")
    counts_table(selected, f"> {SHARE:.0%} of fires at position 0")

    out.append("## Per-matrix component tables\n")
    out.append(
        "Components ordered by share of CI mass at position 0 (descending).  "
        "**mean CI**: the component's average CI over all 2.05M sample tokens; "
        "cell shading is logarithmic over the selected components' full range "
        "(one shared scale for the whole report).  "
        "**Top tokens**: the "
        "component's summed CI per input token id, shown as share of its total "
        "summed CI (up to 8 tokens; tokens below 0.5% share are cut).  **pos 0 … pos 10, rest**: percent of the component's total CI "
        "mass at each chunk position (all CI, not just fires); cell shading is "
        "linear in the percentage.\n")

    all_mci = np.concatenate(
        [zp[f"{m}|Sp"][selected[m]].sum(1) / n_tokens
         for m in mods if len(selected[m])])
    lo, hi = np.log10(all_mci.min()), np.log10(all_mci.max())

    for m in mods:
        ids = selected[m]
        if not len(ids):
            continue
        fp, sp = zp[f"{m}|Fp"], zp[f"{m}|Sp"]
        out.append(f"### `{m}` — {len(ids)} components\n")
        head = ["Component", "fires", "mean CI",
                "top tokens (share of summed CI)"]
        head += [f"pos {p}" for p in range(NPOS)] + ["rest"]
        rows = ['<table style="border-collapse:collapse">',
                "<thead><tr>"
                + "".join(f'<th style="padding:2px 6px;text-align:'
                          f'{"left" if i < 1 or i == 3 else "right"}">{h}</th>'
                          for i, h in enumerate(head))
                + "</tr></thead>", "<tbody>"]
        p0_share = sp[ids, 0] / sp[ids].sum(1)
        for i in ids[np.argsort(-p0_share)]:
            col = ci_tok[:, t_cols[(m, int(i))]]
            total_ci = col.sum()
            top = np.argsort(-col)[:8]
            toks = []
            for rank, t in enumerate(top):
                sh = col[t] / total_ci
                if col[t] == 0 or (rank >= 1 and sh < 0.005):
                    break
                s = html_mod.escape(repr(tokzr.decode([int(t)]))[1:-1]
                                    .replace("&#x27;", "'"))
                toks.append(f"<b>'{s}'</b>&nbsp;{sh:.0%}")
            mass = sp[i] / sp[i].sum()
            mci = sp[i].sum() / n_tokens
            r = ("<tr>"
                 + plain(f"<code>{m}:{i}</code>")
                 + plain(str(int(fp[i].sum())), "right")
                 + cell(f"{mci:.1e}", (np.log10(mci) - lo) / (hi - lo))
                 + plain(", ".join(toks)))
            for p in range(NPOS):
                r += cell(f"{100 * mass[p]:.1f}", mass[p])
            rest = mass[NPOS:].sum()
            r += cell(f"{100 * rest:.1f}", rest) + "</tr>"
            rows.append(r)
        rows += ["</tbody>", "</table>"]
        out.append("\n".join(rows))
        out.append("")

    path = ROOT / "C-pos-0" / "report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path} ({n_total} components)")


if __name__ == "__main__":
    main()
