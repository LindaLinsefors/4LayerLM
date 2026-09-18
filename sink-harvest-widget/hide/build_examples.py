"""Build the activation-examples widget for the sink decompositions C / D / E.

Output: ../activation_examples.html + sidecar data in hide/data/
(the HTML and hide/ must move together; sidecars are loaded on demand via
injected <script src> tags, which works from file://, like compare-decomps).

Per (decomposition, matrix, alive component): up to 16 top activating windows
over the 4,000 cached Pile rows — 51 tokens of surrounding text with a
per-token CI trace (u8-quantized, green background) and a per-token signed
input-activation trace a_t = x_t·V_c (int8-quantized per component, drawn as
a blue/white/red underline; majority-positive sign gauge from
compare-decomps/hide/cache/act_signs_{C,D,E}.npz). Data computed on Modal by
examples_compute_modal.py -> hide/cache/examples_{C,D,E}.npz and
examples_act_modal.py -> hide/cache/examples_act_{C,D,E}.npz
(⚠ public-loader broken-RoPE caveat, sink-models/rope_report.md).

The component header additionally shows the co-authors' 10M-token HARVEST
stats for the selected component (harvest.sqlite of the task-1423 bundles):
harvest mean CI, firing count, eligible regions, examples stored.

Alive = sample mean CI > 1e-6 (coci_{C,D,E}.npz), the established alive proxy
for these runs; the component dropdown contains alive components only,
sorted by sample mean CI descending.

Run: python sink-harvest-widget/hide/build_examples.py   (~2 min)
"""

import base64
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = HERE / "cache"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

RUNS = {"C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35"}
RUN_DESC = {
    "C": "p-d60af588, sink seed 45 (t-87f91319), decomposition seed 0",
    "D": "p-fecd6a6b, sink seed 45 (t-87f91319), decomposition seed 1",
    "E": "p-bd411e35, sink seed 46 (t-75f6c439), decomposition seed 0",
}
SITES = [f"h.{l}.{m}" for l in range(4)
         for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                   "mlp.c_fc", "mlp.down_proj")]
N_WIN, W = 16, 51


def b64(a: np.ndarray) -> str:
    return base64.b64encode(a.tobytes()).decode()


def comp_signs(acts, mod):
    """Per-component sign gauge (+1/-1, full C axis): positive input
    activation on the majority of CI>0.1 tokens; ties broken by the summed
    firing activation; components that never fire by the all-token sum;
    still-ambiguous ones +1. (Copy of coci-heatmaps/hide/interactive_cross.py
    comp_signs — the project-standard gauge.)"""
    Npos, F = acts[f"{mod}|Npos"], acts[f"{mod}|F"]
    maj = np.where(2 * Npos > F, 1.0,
                   np.where(2 * Npos < F, -1.0, np.sign(acts[f"{mod}|Ssum"])))
    s = np.where(F > 0, maj, np.sign(acts[f"{mod}|Sall"]))
    s[s == 0] = 1.0
    return s


def harvest_db(run_id: str) -> Path:
    d = ROOT / "sink-models" / "runs" / run_id / "harvest"
    (sub,) = list(d.iterdir())
    return sub / "harvest.sqlite"


def short_tok(tokenizer, tid: int) -> str:
    t = repr(tokenizer.decode([int(tid)]))
    return t if len(t) <= 18 else t[:15] + "…" + t[0]


def main():
    from load import load_tokenizer
    tokenizer = load_tokenizer("pile_4l")

    print("vocab strings ...")
    vocab = [tokenizer.decode([i]) for i in range(50277)]

    rows = torch.load(ROOT / "context-loss" / "hide" / "cache"
                      / "pile_rows.pt")
    rows = torch.stack([r[:512] for r in rows]).numpy().astype(np.uint16)
    (DATA / "rows.js").write_text(
        '__reg("rows","' + b64(rows) + '");', encoding="utf-8")
    print(f"wrote {DATA / 'rows.js'}")

    meta = {}
    for name, run_id in RUNS.items():
        print(f"--- {name}")
        ex = np.load(CACHE / f"examples_{name}.npz")
        exa = np.load(CACHE / f"examples_act_{name}.npz")
        signs = np.load(ROOT / "compare-decomps" / "hide" / "cache"
                        / f"act_signs_{name}.npz")
        coci = np.load(ROOT / "coci-heatmaps" / "hide" / "cache"
                       / f"coci_{name}.npz")
        top = np.load(ROOT / "mean-ci-widget" / "hide" / "cache"
                      / f"top_tokens_{name}.npz")
        con = sqlite3.connect(f"file:{harvest_db(run_id)}?mode=ro", uri=True)
        meta[name] = {}
        for site in SITES:
            comps = ex[f"{site}|comps"].astype(int)
            mean = coci[f"{site}|mean"].astype(float)
            alive = np.nonzero(mean > 1e-6)[0]
            assert np.array_equal(np.sort(comps), alive), site
            order = np.argsort(-mean[comps], kind="stable")
            # sidecar blob in mean-CI-desc order
            nwin = ex[f"{site}|nwin"][order]
            row = ex[f"{site}|row"][order]
            pos = ex[f"{site}|pos"][order]
            ws = ex[f"{site}|wstart"][order]
            tr = ex[f"{site}|trace"][order]
            tr8 = np.clip(np.nan_to_num(tr.astype(np.float32)) * 255,
                          0, 255).astype(np.uint8)
            # signed activation traces: gauge-fix, per-comp int8 quantization
            act = np.nan_to_num(
                exa[f"{site}|act"][order].astype(np.float32))  # (n, 16, 51)
            act *= comp_signs(signs, site)[comps[order], None, None]
            # zero out unused window slots (their stored trace is 0 already,
            # but be explicit): slots with row == 65535
            act[row == 65535] = 0.0
            scale = np.abs(act).max(axis=(1, 2))          # (n,)
            a8 = np.rint(act / np.maximum(scale, 1e-12)[:, None, None]
                         * 127).astype(np.int8)
            blob = (row.astype("<u2").tobytes() + pos.astype("<u2").tobytes()
                    + ws.astype("<u2").tobytes() + tr8.tobytes()
                    + a8.tobytes())
            (DATA / f"ex_{name}_{site}.js").write_text(
                f'__reg("ex|{name}|{site}",{{"nwin":'
                + json.dumps(nwin.tolist())
                + ',"scale":' + json.dumps(
                    [float(f"{v:.4g}") for v in scale])
                + ',"blob":"' + base64.b64encode(blob).decode() + '"});',
                encoding="utf-8")
            # header metadata (main page)
            hstats = {r[0]: r[1:] for r in con.execute(
                "SELECT component_idx, mean_causal_importance, firing_count,"
                " eligible_region_count, example_count FROM components"
                f" WHERE site='{site}'")}
            ids, vals = top[f"{site}|top_ids"], top[f"{site}|top_ci"]
            tot = top[f"{site}|total"]
            m_l, hint_l, h_l, id_l = [], [], [], []
            for c in comps[order]:
                id_l.append(int(c))
                m_l.append(float(f"{mean[c]:.4g}"))
                share = (float(vals[c][0]) / float(tot[c])
                         if tot[c] > 0 else 0.0)
                hint_l.append(f"{short_tok(tokenizer, ids[c][0])} "
                              f"{share:.0%}" if share > 0 else "—")
                h = hstats.get(int(c))
                h_l.append([float(f"{h[0]:.4g}"), int(h[1]), int(h[2]),
                            int(h[3])] if h else None)
            meta[name][site] = {"id": id_l, "m": m_l, "hint": hint_l,
                                "h": h_l}
        con.close()

    page = (PAGE
            .replace("__VOCAB__", json.dumps(vocab, ensure_ascii=False)
                     .replace("</", "<\\/"))
            .replace("__META__", json.dumps(meta, separators=(",", ":")))
            .replace("__RUNDESC__", json.dumps(RUN_DESC)))
    out = ROOT / "sink-harvest-widget" / "activation_examples.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    total = sum(f.stat().st_size for f in DATA.iterdir())
    print(f"sidecars: {sum(1 for _ in DATA.iterdir())} files, "
          f"{total / 1e6:.0f} MB")


PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Activation examples — sink decompositions C/D/E</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px 24px;
         color: #1a1a2e; }
  h2 { margin: 0 0 4px 0; font-size: 20px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 10px;
         max-width: 1100px; }
  .warn { color: #a15c00; }
  .controls { display: flex; gap: 18px; align-items: center;
              margin-bottom: 10px; flex-wrap: wrap; }
  .controls label { font-size: 14px; }
  select, input[type=text] { font-size: 13px; padding: 3px 6px; }
  #comp { max-width: 420px; }
  button { font-size: 13px; padding: 3px 10px; }
  #header { font-size: 13px; background: #f4f6fa; border-radius: 6px;
            padding: 8px 12px; margin-bottom: 12px; max-width: 1100px; }
  #header b { font-size: 14px; }
  .ex { margin-bottom: 10px; max-width: 1100px; }
  .exhead { color: #888; font-size: 11px; margin-bottom: 1px; }
  .strip { font-family: ui-monospace, Consolas, monospace; font-size: 13px;
           line-height: 1.75; white-space: pre-wrap; word-break: break-all;
           background: #fcfcfb; border: 1px solid #eee; border-radius: 4px;
           padding: 4px 6px; }
  .strip span { border-radius: 2px 2px 0 0;
                border-bottom: 3px solid transparent; }
  .status { color: #888; font-size: 13px; margin: 20px 0; }
</style>
</head>
<body>
<h2>Activation examples per component — sink decompositions C / D / E</h2>
<div class="sub">
  For each <b>alive</b> component (sample mean CI &gt; 1e-6): its top
  activating windows over the 4,000 cached Pile rows (2.05M tokens), up to 16
  non-overlapping windows of 51 tokens. Token background depth (green) =
  causal importance (CI = clip(preact, 0, 1)); token underline = component
  activation a = x·V (majority-positive sign gauge), blue = negative, white =
  zero, red = positive, scaled to the component's max |a| over its shown
  windows; hover any token for its exact CI and activation.
  <span class="warn">⚠ CI computed through the public JAX loader, whose RoPE
  is broken for the sink targets (sink-models/rope_report.md) —
  token-identity-level behavior is expected to survive, long-range/positional
  detail may not. These are our own sample examples: the co-authors' 10M-token
  harvest example windows are not yet readable (the HarvestReader is not
  public; see examples_bin_format_notes.md).</span>
</div>
<div class="controls">
  <label>Decomposition: <select id="run"></select></label>
  <label>Matrix: <select id="site"></select></label>
  <label>Component: <select id="comp"></select></label>
  <button id="prev">◀ prev</button>
  <button id="next">next ▶</button>
  <label>Filter: <input type="text" id="filter" size="14"
         placeholder="id or token"></label>
</div>
<div id="header"></div>
<div id="examples"><div class="status">loading…</div></div>
<script>
const VOCAB = __VOCAB__;
const META = __META__;
const RUNDESC = __RUNDESC__;
const N_WIN = 16, W = 51;

const REG = {}, PENDING = {};
function __reg(key, val) {
  REG[key] = val;
  if (PENDING[key]) { PENDING[key].forEach(f => f(val)); delete PENDING[key]; }
}
function loadSidecar(key, file) {
  if (REG[key] !== undefined) return Promise.resolve(REG[key]);
  if (!PENDING[key]) {
    PENDING[key] = [];
    const s = document.createElement("script");
    s.src = "hide/data/" + file;
    s.onerror = () => __reg(key, null);
    document.head.appendChild(s);
  }
  return new Promise(res => PENDING[key].push(res));
}
function b64bytes(s) {
  const bin = atob(s), a = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
  return a;
}

const runSel = document.getElementById("run");
const siteSel = document.getElementById("site");
const compSel = document.getElementById("comp");
const filterBox = document.getElementById("filter");
for (const r of Object.keys(META)) {
  const o = document.createElement("option");
  o.value = r; o.textContent = r + " — " + RUNDESC[r];
  runSel.appendChild(o);
}
for (const s of Object.keys(META[Object.keys(META)[0]])) {
  const o = document.createElement("option");
  o.value = o.textContent = s;
  siteSel.appendChild(o);
}

function fillComps() {
  const m = META[runSel.value][siteSel.value];
  const f = filterBox.value.trim().toLowerCase();
  const cur = compSel.value;
  compSel.innerHTML = "";
  for (let i = 0; i < m.id.length; i++) {
    const label = m.id[i] + "  ·  CI " + m.m[i] + "  ·  " + m.hint[i];
    if (f && !label.toLowerCase().includes(f)) continue;
    const o = document.createElement("option");
    o.value = i; o.textContent = label;
    compSel.appendChild(o);
  }
  if ([...compSel.options].some(o => o.value === cur)) compSel.value = cur;
}

let rowsArr = null;
async function getRows() {
  if (rowsArr) return rowsArr;
  const b = b64bytes(await loadSidecar("rows", "rows.js"));
  rowsArr = new Uint16Array(b.buffer);
  return rowsArr;
}

function esc(t) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
}

async function render() {
  const run = runSel.value, site = siteSel.value;
  const m = META[run][site];
  const i = parseInt(compSel.value);
  const hdr = document.getElementById("header");
  const holder = document.getElementById("examples");
  if (isNaN(i)) {
    hdr.innerHTML = ""; holder.innerHTML =
      '<div class="status">no component matches the filter</div>';
    return;
  }
  const id = m.id[i];
  let h = "<b>" + site + ":" + id + "</b>  (" + run + ")" +
    " &nbsp;·&nbsp; sample mean CI <b>" + m.m[i] + "</b> (rank " + (i + 1) +
    " of " + m.id.length + " alive)" +
    " &nbsp;·&nbsp; top token " + esc(m.hint[i]);
  if (m.h[i]) {
    h += "<br>harvest (10M tokens): mean CI " + m.h[i][0] +
      " · fires " + m.h[i][1].toLocaleString() +
      " · eligible regions " + m.h[i][2].toLocaleString() +
      " · examples stored " + m.h[i][3];
  } else {
    h += "<br>harvest: no record";
  }
  hdr.innerHTML = h;
  holder.innerHTML = '<div class="status">loading…</div>';

  const key = "ex|" + run + "|" + site;
  const [ex, rows] = await Promise.all([
    loadSidecar(key, "ex_" + run + "_" + site + ".js"), getRows()]);
  if (runSel.value !== run || siteSel.value !== site ||
      parseInt(compSel.value) !== i) return;  // stale render
  const n = ex.nwin.length;
  if (!ex.bytes) ex.bytes = b64bytes(ex.blob);
  const u16 = new Uint16Array(ex.bytes.buffer, 0, n * N_WIN * 3);
  const row = u16.subarray(0, n * N_WIN);
  const pos = u16.subarray(n * N_WIN, 2 * n * N_WIN);
  const ws = u16.subarray(2 * n * N_WIN, 3 * n * N_WIN);
  const tr = new Uint8Array(ex.bytes.buffer, n * N_WIN * 6, n * N_WIN * W);
  const ac = new Int8Array(ex.bytes.buffer, n * N_WIN * 6 + n * N_WIN * W);

  holder.innerHTML = "";
  const nw = ex.nwin[i];
  if (nw === 0) {
    holder.innerHTML = '<div class="status">no firing window found in the ' +
      "sample (component is alive via tiny sub-threshold CI only)</div>";
    return;
  }
  for (let w = 0; w < nw; w++) {
    const r = row[i * N_WIN + w], p = pos[i * N_WIN + w],
          s0 = ws[i * N_WIN + w];
    const trace = tr.subarray((i * N_WIN + w) * W, (i * N_WIN + w + 1) * W);
    const atr = ac.subarray((i * N_WIN + w) * W, (i * N_WIN + w + 1) * W);
    const div = document.createElement("div");
    div.className = "ex";
    let maxci = 0;
    for (let k = 0; k < W; k++) maxci = Math.max(maxci, trace[k]);
    const head = document.createElement("div");
    head.className = "exhead";
    head.textContent = "#" + (w + 1) + " · row " + r + ", pos " + p +
      " · max CI " + (maxci / 255).toFixed(2);
    div.appendChild(head);
    const strip = document.createElement("div");
    strip.className = "strip";
    for (let k = 0; k < W; k++) {
      const tid = rows[r * 512 + s0 + k];
      const ci = trace[k] / 255;
      const an = atr[k] / 127;                    // in [-1, 1]
      const sp = document.createElement("span");
      sp.textContent = VOCAB[tid];
      if (ci > 0)
        sp.style.background = "rgba(31,150,71," + (ci * 0.85).toFixed(3) + ")";
      if (ci > 0.6) sp.style.color = "#fff";
      const lo = Math.round(255 * (1 - Math.abs(an)));
      sp.style.borderBottomColor = an >= 0 ?
        "rgb(255," + lo + "," + lo + ")" : "rgb(" + lo + "," + lo + ",255)";
      sp.title = JSON.stringify(VOCAB[tid]) + " · CI " + ci.toFixed(2) +
        " · a " + (an * ex.scale[i]).toFixed(2);
      strip.appendChild(sp);
    }
    div.appendChild(strip);
    holder.appendChild(div);
  }
}

function onSel() { fillComps(); render(); }
runSel.addEventListener("change", onSel);
siteSel.addEventListener("change", onSel);
compSel.addEventListener("change", render);
filterBox.addEventListener("input", () => { fillComps(); render(); });
document.getElementById("prev").addEventListener("click", () => {
  if (compSel.selectedIndex > 0) { compSel.selectedIndex--; render(); }
});
document.getElementById("next").addEventListener("click", () => {
  if (compSel.selectedIndex < compSel.options.length - 1) {
    compSel.selectedIndex++; render();
  }
});
fillComps();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
