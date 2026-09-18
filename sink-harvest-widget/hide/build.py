"""Build the harvest-statistics interactive widget for the sink decompositions.

One self-contained HTML (../harvest_widget.html, plotly.js inlined) covering the
three 10M-token harvests published 2026-09-17 for decompositions C / D / E
(runs p-d60af588 / p-fecd6a6b / p-bd411e35; see models_and_decomps.md).

Data sources
- harvest.sqlite of each bundle (sink-models/runs/<run>/harvest/<harvest-id>/):
  per-component state (zero_weight / eligible_not_observed / observed),
  firing_count (lower-leaky CI > 0.1 over 10,002,432 corpus tokens),
  mean_causal_importance, mean_activation, mean_absolute_activation,
  maximum_absolute_activation, eligible_region_count, example_count.
  Verified: the stored means equal the parquet sums / 10,002,432.
- our own sample-mean CI over the 4,000 cached Pile rows (2.05M tokens) from
  coci-heatmaps/hide/cache/coci_{C,D,E}.npz, for comparison.  ⚠ These sample
  values were computed through the public JAX loader, whose RoPE is broken for
  the sink targets (see sink-models/rope_report.md) — the harvest presumably
  ran on the co-authors' internal (correct-RoPE?) code, so the scatter view
  doubles as a probe of how much that mismatch matters per component.
- top activating tokens per component (hover) from
  mean-ci-widget/hide/cache/top_tokens_{C,D,E}.npz (same RoPE caveat).

Views (radio):
- "sorted curves": per matrix, components ranked by harvest mean CI desc;
  two lines: harvest mean CI and sample mean CI at the same ranks.
- "harvest vs sample": log-log scatter of the two means, y = x reference,
  Spearman rho in the panel title.

Runs on the default Python 3.11 in ~30 s; extracted data cached in
cache/data.json (delete to re-extract).
"""

import html
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

RUNS = {  # widget name -> (run id, target description)
    "C": ("p-d60af588", "sink seed 45 (t-87f91319), decomposition seed 0"),
    "D": ("p-fecd6a6b", "sink seed 45 (t-87f91319), decomposition seed 1"),
    "E": ("p-bd411e35", "sink seed 46 (t-75f6c439), decomposition seed 0"),
}
N_TOKENS = 10_002_432  # 2,442 batches x 8 x 512

ATTN_MODS = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]
MLP_MODS = ["mlp.c_fc", "mlp.down_proj"]


def harvest_db(run_id: str) -> Path:
    d = ROOT / "sink-models" / "runs" / run_id / "harvest"
    (sub,) = list(d.iterdir())  # exactly one harvest id per bundle
    return sub / "harvest.sqlite"


def wrap(text: str, width: int = 65) -> str:
    return "<br>".join(textwrap.wrap(html.escape(text, quote=False), width))


def top_token_label(tokenizer, ids, vals, total) -> str:
    """Same convention as mean-ci-widget: each token's share of the component's
    summed sample CI; up to 8 tokens, cut below 0.5% share after the first 3."""
    if total <= 0:
        return "(no CI in sample)"
    parts = []
    for tid, v in zip(ids, vals):
        share = float(v) / total
        if v <= 0 or len(parts) >= 8 or (len(parts) >= 3 and share < 0.005):
            break
        tok = repr(tokenizer.decode([int(tid)]))
        if len(tok) > 24:
            tok = tok[:21] + "…" + tok[0]
        pct = f"{share:.1%}" if share < 0.095 else f"{share:.0%}"
        parts.append(f"{tok} {pct}")
    return "top CI tokens: " + " · ".join(parts)


def sig(x: float, n: int = 4) -> float:
    return float(f"{x:.{n}g}")


def extract_run(run_id: str, coci, top, tokenizer) -> dict:
    con = sqlite3.connect(f"file:{harvest_db(run_id)}?mode=ro", uri=True)
    data = {}
    for layer in range(4):
        for block, mods in [("Attn", ATTN_MODS), ("MLP", MLP_MODS)]:
            panels = []
            for mod in mods:
                site = f"h.{layer}.{mod}"
                rows = con.execute(
                    "SELECT component_idx, state, firing_count,"
                    " mean_causal_importance, mean_activation,"
                    " mean_absolute_activation, maximum_absolute_activation,"
                    " eligible_region_count, example_count FROM components"
                    f" WHERE site='{site}' ORDER BY component_idx").fetchall()
                hmean = np.array([r[3] for r in rows])
                smean = coci[f"{site}|mean"].astype(float)
                assert len(hmean) == len(smean)
                ids, vals = top[f"{site}|top_ids"], top[f"{site}|top_ci"]
                total = top[f"{site}|total"]
                order = np.lexsort((np.arange(len(hmean)), -hmean))
                both = (hmean > 0) & (smean > 0)
                rho = (spearmanr(hmean[both], smean[both]).statistic
                       if both.sum() >= 3 else float("nan"))
                states = [r[1] for r in rows]
                panels.append({
                    "name": site,
                    "C": len(rows),
                    "n_obs": states.count("observed"),
                    "n_eligible": states.count("eligible_not_observed"),
                    "n_zero": states.count("zero_weight"),
                    "n_alive_h": int((hmean > 1e-6).sum()),
                    "rho": None if np.isnan(rho) else round(float(rho), 4),
                    "idx": [rows[i][0] for i in order],
                    "h": [sig(hmean[i]) for i in order],
                    "s": [sig(smean[i]) for i in order],
                    "fc": [rows[i][2] for i in order],
                    "erc": [rows[i][7] for i in order],
                    "ec": [rows[i][8] for i in order],
                    "ma": [sig(rows[i][4]) for i in order],
                    "mb": [sig(rows[i][5]) for i in order],
                    "mx": [sig(rows[i][6]) for i in order],
                    "lab": [wrap(top_token_label(tokenizer, ids[i], vals[i],
                                                 float(total[i])))
                            for i in order],
                })
            data[f"L{layer} {block}"] = panels
    con.close()
    return data


def extract() -> dict:
    cache = CACHE / "data.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from load import load_tokenizer
    tokenizer = load_tokenizer("pile_4l")
    data = {}
    for name, (run_id, _) in RUNS.items():
        print(f"extracting {name} ({run_id}) ...")
        coci = np.load(ROOT / "coci-heatmaps" / "hide" / "cache"
                       / f"coci_{name}.npz")
        top = np.load(ROOT / "mean-ci-widget" / "hide" / "cache"
                      / f"top_tokens_{name}.npz")
        data[name] = extract_run(run_id, coci, top, tokenizer)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Harvest statistics — sink decompositions C/D/E</title>
<script>__PLOTLYJS__</script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px 24px; color: #1a1a2e; }
  h2 { margin: 0 0 4px 0; font-size: 20px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 10px; max-width: 1100px; }
  .warn { color: #a15c00; }
  .controls { display: flex; gap: 22px; align-items: center; margin-bottom: 8px;
               flex-wrap: wrap; }
  .controls label { font-size: 14px; }
  select { font-size: 14px; padding: 3px 6px; }
  .note { color: #888; font-size: 12px; max-width: 620px; }
</style>
</head>
<body>
<h2>Harvest statistics per component — sink decompositions C / D / E</h2>
<div class="sub">
  10M-token harvests (task-1423 bundles, published 2026-09-17): 2,442 batches
  &times; 8 &times; 512 tokens; firing = lower-leaky CI &gt; 0.1.
  <b>harvest mean CI</b> = mean_causal_importance from harvest.sqlite
  (sum over all 10,002,432 tokens / 10,002,432).
  <b>sample mean CI</b> = our own mean over the 4,000 cached Pile rows
  (2.05M tokens).
  <span class="warn">⚠ the sample series (and the hover top-tokens) were
  computed through the public JAX loader, whose RoPE is broken for the sink
  targets (sink-models/rope_report.md); the harvest presumably ran on internal
  code — the scatter view doubles as a per-component probe of that
  mismatch.</span>
</div>
<div class="controls">
  <label>Decomposition: <select id="run"></select></label>
  <label>Matrix block: <select id="block"></select></label>
  <label>View: <select id="view">
    <option value="curves">sorted curves</option>
    <option value="scatter">harvest vs sample</option>
  </select></label>
  <label><input type="radio" name="yscale" value="log" checked> log</label>
  <label><input type="radio" name="yscale" value="linear"> linear</label>
  <span class="note" id="viewnote"></span>
</div>
<div id="plots"></div>
<script>
const DATA = __DATA__;
const RUNMETA = __RUNMETA__;
const COL_H = "#2a78d6";   /* harvest mean CI  */
const COL_S = "#1baf7a";   /* sample mean CI   */
const runSel = document.getElementById("run");
const blockSel = document.getElementById("block");
const viewSel = document.getElementById("view");
for (const name of Object.keys(DATA)) {
  const o = document.createElement("option");
  o.value = name;
  o.textContent = name + " — " + RUNMETA[name];
  runSel.appendChild(o);
}
for (const name of Object.keys(DATA[Object.keys(DATA)[0]])) {
  const o = document.createElement("option");
  o.value = o.textContent = name;
  blockSel.appendChild(o);
}

function yscale() {
  return document.querySelector('input[name="yscale"]:checked').value;
}

function hoverBody(p) {
  /* customdata rows: [idx, h, s, fc, erc, ec, ma, mb, mx, lab] */
  return "<b>" + p.name + ":%{customdata[0]}</b>" +
    "<br>harvest mean CI = %{customdata[1]:.3g}" +
    "   ·   sample mean CI = %{customdata[2]:.3g}" +
    "<br>fires %{customdata[3]:,} of 10,002,432 tokens" +
    "   ·   eligible regions %{customdata[4]:,}" +
    "   ·   examples stored %{customdata[5]}" +
    "<br>activation: mean %{customdata[6]:.3g}, mean |a| %{customdata[7]:.3g}," +
    " max |a| %{customdata[8]:.3g}" +
    "<br>%{customdata[9]}<extra></extra>";
}

function customdata(p) {
  return p.idx.map((id, i) => [id, p.h[i], p.s[i], p.fc[i], p.erc[i],
                               p.ec[i], p.ma[i], p.mb[i], p.mx[i], p.lab[i]]);
}

function render() {
  const panels = DATA[runSel.value][blockSel.value];
  const view = viewSel.value;
  document.getElementById("viewnote").textContent = view === "curves"
    ? "components sorted by harvest mean CI (descending); dashed green line: " +
      "the 1e-6 alive cutoff. Hover for full per-component harvest stats."
    : "each dot one component (only components with both means > 0); gray " +
      "dashed line y = x; \\u03c1 = Spearman rank correlation. Hover for stats.";
  const holder = document.getElementById("plots");
  holder.innerHTML = "";
  for (const p of panels) {
    const div = document.createElement("div");
    div.style.height = view === "curves" ? "280px" : "420px";
    if (view === "scatter") {
      div.style.maxWidth = "560px";
      div.style.display = "inline-block";
      div.style.width = "48%";
      div.style.minWidth = "420px";
    }
    holder.appendChild(div);
    const cd = customdata(p);
    const counts = p.n_obs + " observed / " + p.n_eligible +
      " eligible-never-fired / " + p.n_zero + " zero-weight";
    let traces, layout;
    if (view === "curves") {
      const x = p.h.map((_, i) => i);
      traces = [
        { x: x, y: p.h, customdata: cd, mode: "lines", name: "harvest mean CI",
          line: { color: COL_H, width: 2 }, hovertemplate: hoverBody(p) },
        { x: x, y: p.s, customdata: cd, mode: "lines", name: "sample mean CI",
          line: { color: COL_S, width: 1.2 }, hovertemplate: hoverBody(p) },
      ];
      layout = {
        title: { text: p.name + " — " + p.n_alive_h +
                 " of " + p.C + " above 1e-6 (harvest)  ·  " + counts,
                 font: { size: 13 }, x: 0, xanchor: "left" },
        margin: { l: 60, r: 20, t: 30, b: 34 },
        xaxis: { title: { text: "component rank (by harvest mean CI)",
                          font: { size: 12 } },
                 showspikes: true, spikemode: "across", spikethickness: 1,
                 spikedash: "dot", zeroline: false },
        yaxis: { title: { text: "mean CI", font: { size: 12 } },
                 type: yscale(), zeroline: false, exponentformat: "power" },
        hovermode: "closest",
        hoverlabel: { align: "left", font: { size: 12 } },
        showlegend: true,
        legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12,
                  font: { size: 12 } },
        shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper",
                   y0: 1e-6, y1: 1e-6,
                   line: { color: "#2e8b57", width: 1.5, dash: "dash" } }],
      };
    } else {
      const xs = [], ys = [], cds = [];
      for (let i = 0; i < p.h.length; i++)
        if (p.h[i] > 0 && p.s[i] > 0) {
          xs.push(p.s[i]); ys.push(p.h[i]); cds.push(cd[i]);
        }
      traces = [{
        x: xs, y: ys, customdata: cds, mode: "markers",
        marker: { color: COL_H, size: 4, opacity: 0.55 },
        hovertemplate: hoverBody(p),
      }];
      const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
      layout = {
        title: { text: p.name + "  ·  \\u03c1 = " +
                 (p.rho === null ? "n/a" : p.rho) + "  ·  n = " + xs.length,
                 font: { size: 13 }, x: 0, xanchor: "left" },
        margin: { l: 64, r: 16, t: 30, b: 44 },
        xaxis: { title: { text: "sample mean CI (broken-RoPE loader)",
                          font: { size: 12 } },
                 type: yscale(), zeroline: false, exponentformat: "power" },
        yaxis: { title: { text: "harvest mean CI", font: { size: 12 } },
                 type: yscale(), zeroline: false, exponentformat: "power" },
        hovermode: "closest",
        hoverlabel: { align: "left", font: { size: 12 } },
        showlegend: false,
        shapes: [{ type: "line", x0: lo, y0: lo, x1: hi, y1: hi,
                   line: { color: "#999", width: 1, dash: "dash" } }],
      };
    }
    Plotly.newPlot(div, traces, layout,
                   { responsive: true, displayModeBar: false });
  }
}

runSel.addEventListener("change", render);
blockSel.addEventListener("change", render);
viewSel.addEventListener("change", render);
for (const r of document.querySelectorAll('input[name="yscale"]'))
  r.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def main():
    data = extract()
    from plotly.offline import get_plotlyjs
    meta = {name: f"{run_id}, {desc}" for name, (run_id, desc) in RUNS.items()}
    page = (PAGE
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__RUNMETA__", json.dumps(meta)))
    out = ROOT / "sink-harvest-widget" / "harvest_widget.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
