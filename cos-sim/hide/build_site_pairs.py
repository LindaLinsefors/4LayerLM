"""Build cos-sim/site_pairs_interactive.html — interactive version of the
static vcos_site_pairs.png / ucos_site_pairs.png figures in the signed
reports: per-site-pair counts of component pairs with signed cosine above an
adjustable threshold (0.2, 0.3, ..., 0.8), for every combination of
{read-in, write-out} × {newA, C}, plus a toggle to count the negative tail
(cos < -thr) instead.

Vectors and gauge exactly as in the two compute_signed.py scripts: gauge-fixed
unit read-ins V (read-in; 16 matrices q/k/v/c_fc) resp. unit write directions
U (write-out; 8 matrices o/down), alive components only.  Counts for all
site pairs x thresholds x signs are precomputed here and embedded as JSON;
plotly.js is inlined (self-contained file).

Usage: python cos-sim/hide/build_site_pairs.py   (~2 min, default Python 3.11)
"""
import json
import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "coci-heatmaps" / "hide"))
from interactive_cross import comp_signs  # noqa: E402

CH = ROOT / "coci-heatmaps" / "hide" / "cache"
CD = ROOT / "compare-decomps" / "hide" / "cache"

THRS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

SIDES = {
    "read-in": dict(
        fac="V",
        sites=[f"h.{l}.{m}" for l in range(4)
               for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.c_fc")],
        short=[f"L{l}{m}" for l in range(4) for m in ("q", "k", "v", "c_fc")],
        group=4),
    "write-out": dict(
        fac="U",
        sites=[f"h.{l}.{m}" for l in range(4)
               for m in ("attn.o_proj", "mlp.down_proj")],
        short=[f"L{l}{m}" for l in range(4) for m in ("o", "down")],
        group=2),
}

DECOMPS = {
    "newA": dict(uv=CH / "uv_newA.npz", acts=CH / "act_signs_newA.npz"),
    "C": dict(uv=CD / "uv_C.npz", acts=CD / "act_signs_C.npz"),
}


def alive_ids(name, site):
    if name == "newA":  # uv_newA rows are exactly these ids, in this order
        return np.load(CH / "cross_alive.npz")[f"newA|{site}"].astype(np.int32)
    cc = np.load(CH / f"coci_{name}.npz")
    return np.flatnonzero(cc[f"{site}|mean"] > 1e-6).astype(np.int32)


def site_vectors(side, name):
    """Gauge-fixed unit vectors per site, exactly as compute_signed.py pools them."""
    cfg = SIDES[side]
    uv = np.load(DECOMPS[name]["uv"])
    acts = np.load(DECOMPS[name]["acts"])
    out = []
    for site in cfg["sites"]:
        alive = alive_ids(name, site)
        sign = comp_signs(acts, site)[alive]
        Xall = uv[f"{site}|{cfg['fac']}"]
        X = (Xall if len(Xall) == len(alive) else Xall[alive]).astype(np.float32)
        X = X * sign[:, None]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        out.append(X)
    return out


def combo_counts(side, name):
    vs = site_vectors(side, name)
    S = len(vs)
    pos = np.zeros((len(THRS), S, S), np.int64)
    neg = np.zeros((len(THRS), S, S), np.int64)
    for s1 in range(S):
        for s2 in range(s1, S):
            C = vs[s1] @ vs[s2].T
            if s1 == s2:
                C = C[np.triu_indices(len(C), k=1)]
            for t, thr in enumerate(THRS):
                p = int((C > thr).sum())
                n = int((C < -thr).sum())
                pos[t, s1, s2] = pos[t, s2, s1] = p
                neg[t, s1, s2] = neg[t, s2, s1] = n
    n_comps = [len(v) for v in vs]
    print(f"{side}/{name}: {sum(n_comps)} comps; pairs >+0.7: "
          f"{int(pos[THRS.index(0.7)][np.triu_indices(S)].sum())}")
    return dict(short=SIDES[side]["short"], group=SIDES[side]["group"],
                n=n_comps, pos=pos.tolist(), neg=neg.tolist())


DATA = {f"{side}|{name}": combo_counts(side, name)
        for side in SIDES for name in DECOMPS}

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>cos-sim site-pair counts</title>
<script>__PLOTLYJS__</script>
<style>
body { font-family: Segoe UI, Arial, sans-serif; margin: 14px; color: #333; }
h2 { margin: 0 0 6px 0; font-size: 18px; }
.controls { display: flex; gap: 26px; align-items: center; flex-wrap: wrap;
            margin: 8px 0 4px 0; font-size: 14px; }
.controls label { margin-right: 4px; }
select { font-size: 14px; }
#thrval { font-weight: bold; display: inline-block; width: 2.4em; }
.note { color: #777; font-size: 12px; margin-bottom: 4px; }
</style>
</head>
<body>
<h2>Component pairs per site pair with signed cosine beyond threshold</h2>
<div class="note">Signed cos between gauge-fixed unit read-ins V (read-in:
q/k/v/c_fc) resp. write directions U (write-out: o_proj/down_proj), alive
components, majority-positive-activation gauge — the interactive version of
the vcos/ucos_site_pairs.png figures in the cos-sim reports. Diagonal cells
count within-matrix pairs (each pair once); color = log10(count+1), fixed per
view across thresholds.</div>
<div class="controls">
  <span><label for="combo">Data:</label>
  <select id="combo">
    <option value="read-in|newA">read-in &mdash; new A</option>
    <option value="read-in|C">read-in &mdash; C</option>
    <option value="write-out|newA">write-out &mdash; new A</option>
    <option value="write-out|C">write-out &mdash; C</option>
  </select></span>
  <span>
    <label><input type="radio" name="sign" value="pos" checked> cos &gt; +thr</label>
    <label><input type="radio" name="sign" value="neg"> cos &lt; &minus;thr</label>
  </span>
  <span><label for="thr">threshold:</label>
  <input type="range" id="thr" min="0" max="6" step="1" value="5">
  <span id="thrval">0.7</span></span>
  <span id="total"></span>
</div>
<div id="plot" style="width: 760px; height: 700px;"></div>
<script type="application/json" id="DATA">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("DATA").textContent);
const THRS = __THRS__;

function render() {
  const combo = document.getElementById("combo").value;
  const sign = document.querySelector("input[name=sign]:checked").value;
  const ti = +document.getElementById("thr").value;
  document.getElementById("thrval").textContent = THRS[ti].toFixed(1);
  const d = DATA[combo];
  const M = d[sign][ti];
  const S = d.short.length;
  const z = [], text = [], hover = [];
  let total = 0;
  for (let i = 0; i < S; i++) {
    const zr = [], tr = [], hr = [];
    for (let j = 0; j < S; j++) {
      const v = M[i][j];
      if (j >= i) total += v;
      zr.push(Math.log10(v + 1));
      tr.push(v ? String(v) : "");
      hr.push(`${d.short[i]} (n=${d.n[i]}) × ${d.short[j]} (n=${d.n[j]})` +
              `<br>${v} pairs ${sign === "pos" ? ">" : "< −"}${THRS[ti].toFixed(1)}`);
    }
    z.push(zr); text.push(tr); hover.push(hr);
  }
  // color scale fixed per (combo, sign) at the loosest threshold
  let zmax = 0;
  for (const row of d[sign][0]) for (const v of row)
    zmax = Math.max(zmax, Math.log10(v + 1));
  const sep = [];
  for (let k = d.group; k < S; k += d.group) {
    sep.push({type: "line", x0: k - 0.5, x1: k - 0.5, y0: -0.5, y1: S - 0.5,
              line: {color: "#999", width: 1}});
    sep.push({type: "line", y0: k - 0.5, y1: k - 0.5, x0: -0.5, x1: S - 0.5,
              line: {color: "#999", width: 1}});
  }
  const sgn = sign === "pos" ? "> +" : "< −";
  document.getElementById("total").textContent =
    `total: ${total} pairs ${sgn}${THRS[ti].toFixed(1)}`;
  Plotly.react("plot", [{
    type: "heatmap", z: z, text: text, hovertext: hover,
    hoverinfo: "text", texttemplate: "%{text}",
    textfont: {size: S > 10 ? 9 : 12},
    colorscale: "Blues", reversescale: true, zmin: 0, zmax: Math.max(zmax, 1),
    colorbar: {title: {text: "log10(count+1)", side: "right"}, thickness: 14},
    xgap: 1, ygap: 1,
  }], {
    title: {text: `${combo.replace("|", " — ")}: pairs with cos ${sgn}${THRS[ti].toFixed(1)}`,
            font: {size: 15}},
    xaxis: {tickvals: [...Array(S).keys()], ticktext: d.short, tickangle: 90,
            constrain: "domain"},
    yaxis: {tickvals: [...Array(S).keys()], ticktext: d.short,
            autorange: "reversed", scaleanchor: "x", constrain: "domain"},
    shapes: sep, margin: {t: 40, l: 70, r: 10, b: 80},
  }, {displaylogo: false});
}

document.getElementById("combo").addEventListener("change", render);
document.getElementById("thr").addEventListener("input", render);
for (const el of document.querySelectorAll("input[name=sign]"))
  el.addEventListener("change", render);
render();
</script>
</body>
</html>
"""

from plotly.offline import get_plotlyjs  # noqa: E402

out = (HTML
       .replace("__THRS__", json.dumps(THRS))
       .replace("__DATA__", json.dumps(DATA).replace("</", "<\\/"))
       .replace("__PLOTLYJS__", get_plotlyjs()))
dest = ROOT / "cos-sim" / "site_pairs_interactive.html"
dest.write_text(out, encoding="utf-8")
print(f"written {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
