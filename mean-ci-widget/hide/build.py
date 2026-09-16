"""Build the mean-CI-per-component interactive widgets (one HTML per model).

For every matrix of each model: all C components sorted by mean CI (descending),
drawn as a line; hover shows the component id, mean CI, and its description.
A dropdown picks the layer x block (L0 Attn ... L3 MLP): Attn shows 4 subplots
(q/k/v/o), MLP shows 2 (c_fc/down_proj).

Data sources
- pile_4l:   mean CI from harvest.db (mean_activations JSON, causal_importance);
             descriptions from the paper site's labels.json (all 9,973 alive
             components; static.goodfire.ai/vpd-blog-post/data/model-overview/),
             falling back to the local interp.db label (a different, vaguer
             autointerp run) for non-exported components.
- simple_2l: mean CI from harvest.db (mean_ci column); descriptions from the
             local interp.db.
- newA/newB (the two 800k-step pile decompositions p-8383f5e5 / p-4d9a6a12):
             sample-mean CI over the 4,000 cached Pile rows (2.05M tokens) from
             coci-heatmaps/hide/cache/coci_{newA,newB}.npz — no harvest run or
             autointerp labels exist for these; "alive" (> 1e-6) is a
             sample-mean proxy, and instead of a description the hover lists
             the component's top activating tokens (share of its summed CI per
             input token id, from cache/top_tokens_{newA,newB}.npz — computed
             on Modal by top_tokens_compute_modal.py over the same sample).
- C/D/E (the three decompositions of the attention-sink/untied-head targets;
             see models_and_decomps.md): same data sources as newA/newB, from
             coci_{C,D,E}.npz + top_tokens_{C,D,E}.npz (both computed by
             coci-heatmaps/hide/coci_compute_sink_modal.py in one pass).

Outputs ../mean_ci_{pile_4l,simple_2l,newA,newB,C,D,E}.html (self-contained,
plotly.js inlined). Extracted data cached in cache/data.json; site labels in
cache/site_labels.json. Runs on the default Python 3.11 in ~10 s.
"""

import html
import json
import sqlite3
import textwrap
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / "cache"

MODELS_DATA = ROOT / "prev_paper" / "models"
SITE_LABELS_URL = ("https://static.goodfire.ai/vpd-blog-post/data/"
                   "model-overview/labels.json")

# per-matrix subcomponent counts C, from each decomposition's final_config.yaml
C_PILE = {"attn.q_proj": 512, "attn.k_proj": 512, "attn.v_proj": 1024,
          "attn.o_proj": 1024, "mlp.c_fc": 3072, "mlp.down_proj": 3584}
C_SIMPLE = {"attn.q_proj": 288, "attn.k_proj": 288, "attn.v_proj": 384,
            "attn.o_proj": 480, "mlp.c_fc": 1152, "mlp.down_proj": 960}

ATTN_MODS = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"]
MLP_MODS = ["mlp.c_fc", "mlp.down_proj"]

# harvest module name -> paper-site short name ('h.0.mlp.c_fc:5' -> '0.mlp.up:5')
SITE_SHORT = {"attn.q_proj": "attn.q", "attn.k_proj": "attn.k",
              "attn.v_proj": "attn.v", "attn.o_proj": "attn.o",
              "mlp.c_fc": "mlp.up", "mlp.down_proj": "mlp.down"}


def site_labels() -> dict:
    path = CACHE / "site_labels.json"
    if not path.exists():
        print("fetching", SITE_LABELS_URL)
        with urllib.request.urlopen(SITE_LABELS_URL) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text(encoding="utf-8"))


def load_interp_labels(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = dict(con.execute("SELECT component_key, label FROM interpretations"))
    con.close()
    return out


def load_mean_ci(path: Path, json_column: bool) -> dict:
    """component_key -> mean CI. Pile stores it inside the mean_activations
    JSON; SimpleStories has a plain mean_ci column."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    if json_column:
        rows = con.execute(
            "SELECT component_key, mean_activations FROM components")
        out = {k: json.loads(v)["causal_importance"] for k, v in rows}
    else:
        out = dict(con.execute("SELECT component_key, mean_ci FROM components"))
    con.close()
    return out


def wrap(text: str, width: int = 65) -> str:
    """HTML-escape (labels can contain e.g. '<|endoftext|>', which would break
    the hover HTML and the embedded <script> JSON) and line-wrap for hover."""
    return "<br>".join(textwrap.wrap(html.escape(text, quote=False), width))


def build_model_data(n_layers, c_per_mod, mean_ci, interp, site=None):
    """-> {option -> [per-matrix dicts]}; options 'L0 Attn', 'L0 MLP', ..."""
    data = {}
    for layer in range(n_layers):
        for block, mods in [("Attn", ATTN_MODS), ("MLP", MLP_MODS)]:
            panels = []
            for mod in mods:
                prefix = f"h.{layer}.{mod}"
                comps = []
                for idx in range(c_per_mod[mod]):
                    key = f"{prefix}:{idx}"
                    ci = mean_ci.get(key, 0.0)
                    label = src = None
                    if site is not None:
                        label = site.get(f"{layer}.{SITE_SHORT[mod]}:{idx}")
                        if label is not None:
                            src = "paper-site label"
                    if label is None and key in interp:
                        label, src = interp[key], ("local interp.db label"
                                                   if site else "interp.db label")
                    if label is None:
                        label, src = "(no label)", ("never fired"
                                                    if ci == 0 else "no label")
                    comps.append((ci, idx, wrap(label), src))
                comps.sort(key=lambda t: (-t[0], t[1]))
                panels.append({
                    "name": prefix,
                    "n_alive": sum(1 for c in comps if c[0] > 1e-6),
                    "C": c_per_mod[mod],
                    "ci": [round(c[0], 9) for c in comps],
                    "idx": [c[1] for c in comps],
                    "label": [c[2] for c in comps],
                    "src": [c[3] for c in comps],
                })
            data[f"L{layer} {block}"] = panels
    return data


def top_token_label(tokenizer, ids, vals, total) -> str:
    """'top CI tokens: \\' the\\' 34% · ...' — share of the component's summed
    CI landing on each input token id; up to 8 tokens, stopping early once
    shares drop below 0.5% (after the first 3)."""
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


def build_new_data(coci_path: Path, top_path: Path, tokenizer) -> dict:
    """Per-matrix data for a new (JAX, 800k-step) decomposition: sample-mean CI
    from the coci-heatmaps cache; in place of autointerp labels, each
    component's top activating tokens from the top_tokens cache."""
    z = np.load(coci_path)
    zt = np.load(top_path)
    data = {}
    for layer in range(4):
        for block, mods in [("Attn", ATTN_MODS), ("MLP", MLP_MODS)]:
            panels = []
            for mod in mods:
                prefix = f"h.{layer}.{mod}"
                mean = z[f"{prefix}|mean"].astype(float)
                ids, vals = zt[f"{prefix}|top_ids"], zt[f"{prefix}|top_ci"]
                total = zt[f"{prefix}|total"]
                order = np.lexsort((np.arange(len(mean)), -mean))
                panels.append({
                    "name": prefix,
                    "n_alive": int((mean > 1e-6).sum()),
                    "C": len(mean),
                    "ci": [round(float(mean[i]), 9) for i in order],
                    "idx": [int(i) for i in order],
                    "label": [wrap(top_token_label(tokenizer, ids[i], vals[i],
                                                   float(total[i])))
                              for i in order],
                    "src": ["top tokens over the 2.05M-token sample "
                            "(no autointerp)"] * len(mean),
                })
            data[f"L{layer} {block}"] = panels
    return data


def extract() -> dict:
    cache = CACHE / "data.json"
    data = (json.loads(cache.read_text(encoding="utf-8"))
            if cache.exists() else {})
    changed = False

    if "pile_4l" not in data:
        pile_dir = MODELS_DATA / "pile_4layer" / "additional-component-data"
        print("reading pile harvest.db ...")
        data["pile_4l"] = build_model_data(
            4, C_PILE,
            load_mean_ci(pile_dir / "harvest.db", json_column=True),
            load_interp_labels(pile_dir / "interp.db"),
            site=site_labels())
        changed = True
    if "simple_2l" not in data:
        simple_dir = (MODELS_DATA / "simplestories_2layer"
                      / "additional-component-data")
        print("reading simplestories harvest.db ...")
        data["simple_2l"] = build_model_data(
            2, C_SIMPLE,
            load_mean_ci(simple_dir / "harvest" / "s-eab2ace8"
                         / "h-20260212_142914" / "harvest.db",
                         json_column=False),
            load_interp_labels(simple_dir / "autointerp" / "s-eab2ace8"
                               / "a-20260212_142914" / "interp.db"))
        changed = True
    if not {"newA", "newB", "C", "D", "E"} <= data.keys():
        import sys
        sys.path.insert(0, str(ROOT))
        from load import load_tokenizer
        tokenizer = load_tokenizer("pile_4l")
        for name in ("newA", "newB", "C", "D", "E"):
            if name not in data:
                print(f"reading coci_{name}.npz + top_tokens_{name}.npz ...")
                data[name] = build_new_data(
                    ROOT / "coci-heatmaps" / "hide" / "cache"
                    / f"coci_{name}.npz",
                    CACHE / f"top_tokens_{name}.npz", tokenizer)
                changed = True

    if changed:
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mean CI per component — __TITLE__</title>
<script>__PLOTLYJS__</script>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px 24px; color: #1a1a2e; }
  h2 { margin: 0 0 4px 0; font-size: 20px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 12px; }
  .controls { display: flex; gap: 24px; align-items: center; margin-bottom: 8px;
               flex-wrap: wrap; }
  .controls label { font-size: 14px; }
  select { font-size: 14px; padding: 3px 6px; }
  .note { color: #888; font-size: 12px; }
</style>
</head>
<body>
<h2>Mean causal importance per component — __TITLE__</h2>
<div class="sub">__SUBTITLE__</div>
<div class="controls">
  <label>Matrix block:
    <select id="block"></select>
  </label>
  <label><input type="radio" name="yscale" value="log" checked> log y</label>
  <label><input type="radio" name="yscale" value="linear"> linear y</label>
  <span class="note">components sorted by mean CI (descending); never-fired
    components have mean CI = 0 and only appear on the linear scale.
    Hover the line for each component's details. Dashed green line:
    alive&thinsp;|&thinsp;dead cutoff (mean CI = 1e-6).</span>
</div>
<div id="plots"></div>
<script>
const DATA = __DATA__;
const LINE = "#4269d0";
const blockSel = document.getElementById("block");
for (const name of Object.keys(DATA)) {
  const o = document.createElement("option");
  o.value = o.textContent = name;
  blockSel.appendChild(o);
}

function yscale() {
  return document.querySelector('input[name="yscale"]:checked').value;
}

function render() {
  const panels = DATA[blockSel.value];
  const holder = document.getElementById("plots");
  holder.innerHTML = "";
  for (const p of panels) {
    const div = document.createElement("div");
    div.style.height = "270px";
    holder.appendChild(div);
    const hasLabels = "label" in p;
    const trace = {
      x: p.ci.map((_, i) => i),
      y: p.ci,
      customdata: hasLabels ? p.idx.map((id, i) => [id, p.label[i], p.src[i]])
                            : p.idx.map(id => [id]),
      mode: "lines",
      line: { color: LINE, width: 2 },
      hovertemplate:
        "<b>" + p.name + ":%{customdata[0]}</b>  (rank %{x})<br>" +
        "mean CI = %{y:.3g}" +
        (hasLabels ? "<br>%{customdata[1]}" +
                     "<br><span style='color:#888'>%{customdata[2]}</span>"
                   : "") +
        "<extra></extra>",
    };
    const layout = {
      title: { text: p.name + " — " + p.n_alive + " alive of " + p.C +
               " components", font: { size: 14 }, x: 0, xanchor: "left" },
      margin: { l: 60, r: 20, t: 34, b: 34 },
      xaxis: { title: { text: "component rank (by mean CI)", font: { size: 12 } },
               showspikes: true, spikemode: "across", spikethickness: 1,
               spikedash: "dot", zeroline: false },
      yaxis: { title: { text: "mean CI", font: { size: 12 } },
               type: yscale(), zeroline: false, exponentformat: "power" },
      hovermode: "x",
      hoverlabel: { align: "left", font: { size: 12 } },
      showlegend: false,
      shapes: [{
        type: "line", x0: 0, x1: 1, xref: "paper",
        y0: 1e-6, y1: 1e-6,
        line: { color: "#2e8b57", width: 1.5, dash: "dash" },
      }],
    };
    Plotly.newPlot(div, [trace], layout,
                   { responsive: true, displayModeBar: false });
  }
}

blockSel.addEventListener("change", render);
for (const r of document.querySelectorAll('input[name="yscale"]'))
  r.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


def write_html(model_key, title, subtitle, data):
    from plotly.offline import get_plotlyjs
    html = (PAGE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__DATA__", json.dumps(data[model_key],
                                            separators=(",", ":"))))
    out = ROOT / "mean-ci-widget" / f"mean_ci_{model_key}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    data = extract()
    write_html(
        "pile_4l", "pile_4l (4-layer Pile model)",
        "Mean CI over the harvest run (20,000 batches &times; 32 seqs); "
        "descriptions from the paper site's autointerp labels where exported "
        "(alive components), otherwise from the local interp.db run.",
        data)
    write_html(
        "simple_2l", "simple_2l (2-layer SimpleStories model)",
        "Mean CI over the harvest run (20,000 batches &times; 32 seqs); "
        "descriptions from the local interp.db autointerp run.",
        data)
    new_sub = ("Mean CI = sample mean over the 4,000 cached Pile rows "
               "(2.05M tokens); no harvest run or autointerp labels exist for "
               "this decomposition, so “alive” (mean CI &gt; 1e-6) is a "
               "sample-mean proxy and, in place of a description, hover lists "
               "the component's top activating tokens (each token's share of "
               "the component's total CI over the sample).")
    write_html(
        "newA", "new A (pile_4l decomposition p-8383f5e5, 800k steps)",
        new_sub, data)
    write_html(
        "newB", "new B (pile_4l decomposition p-4d9a6a12, 800k steps)",
        new_sub, data)
    write_html(
        "C", "C (decomposition p-d60af588 of sink seed 45 t-87f91319, seed 0)",
        new_sub, data)
    write_html(
        "D", "D (decomposition p-fecd6a6b of sink seed 45 t-87f91319, seed 1)",
        new_sub, data)
    write_html(
        "E", "E (decomposition p-bd411e35 of sink seed 46 t-75f6c439, seed 0)",
        new_sub, data)


if __name__ == "__main__":
    main()
