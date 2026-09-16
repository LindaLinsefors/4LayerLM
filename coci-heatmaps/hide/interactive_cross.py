"""Interactive cross-decomposition heatmaps for ALL 24 matrices — any of
old / newA / newB on either axis.

One self-contained HTML (plotly.js inlined): two dropdowns pick the y- and
x-axis decomposition (9 combos), a third picks the matrix, a radio picks the
measure — co-CI r / cos(U) / cos(V). The cosines are SIGNED: each component's
(U_c, V_c) sign gauge is fixed so that its input activation V_c . x is
positive on the majority of tokens where it is causally important (CI > 0.1;
majority of firing tokens, ties by the summed activation, never-firing
components by the all-token sum) — statistics over the 4,000 cached Pile rows
from cache/act_signs_{old,newA,newB}.npz (act_signs_modal.py). When the two
decompositions differ,
the axes use the static reports' diagonal-matched ordering (y by descending
mean CI, each x component at its best co-CI-match row; the match-threshold
checkbox toggles the r >= MATCH_R unmatched right tail exactly as before).
When they are the SAME decomposition, both axes are simply sorted by
descending mean CI and the checkbox is disabled.

Hovering a cell shows the two component ids + the value in the tooltip, and
fills an info panel with all three measures for that pair plus, per
component: mean CI, its top activating tokens (newA/newB; mean-ci-widget
top_tokens caches) or its autointerp label (old; mean-ci-widget data.json),
and a "pos-0 component" flag (> 50% of its CI > 0.1 fires at chunk position
0, min 20 fires; from cache/pos_fires_{old,newA,newB}.npz). Pos-0 components
are also marked with green edge ticks (left = y rows, top = x columns) that
track cells on zoom.

Size: ~93M cells x 3 measures is far too much for plain JSON, so the
matrices are int8-quantized (value*127; -128 = undefined; ~0.008 error, hover
shows 2 decimals), zlib-compressed and base64-embedded per (pair, matrix) —
cross pairs stored once with both axis orders as permutations (a swap costs
nothing), same-decomposition matrices as the upper triangle (symmetric); the
page decodes them lazily with DecompressionStream("deflate") on first
selection. Each blob sits in its OWN inert <script type="text/plain"> tag,
NOT inside the startup JSON: parsing ~180 MB of base64 through JSON.parse at
load (and holding it as JS strings forever) is what made the previous
revision slow to open — now the startup JSON is only ~13 MB (meta/ids/perms)
and a blob's text is pulled from the DOM on first use.

Writes old-newA-newB/interactive_newA_newB.html.

Usage: python coci-heatmaps/hide/interactive_cross.py   (~10 min)
"""

import base64
import html as html_mod
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
ROOT = HERE.parent.parent
OUT = HERE.parent / "old-newA-newB" / "interactive_newA_newB.html"
MCW_CACHE = ROOT / "mean-ci-widget" / "hide" / "cache"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from report_cross import (MODS, ci_orders, cross_stats,  # noqa: E402
                          pair_r, MATCH_R)
from load import load_tokenizer  # noqa: E402

DEC = ["old", "newA", "newB"]
CROSS_PAIRS = [("old", "newA"), ("old", "newB"), ("newA", "newB")]


def unit(rows):
    x = rows.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = np.inf
    return x / n


def q8(a):
    """float array in [-1, 1] -> int8 (value*127), NaN -> -128."""
    q = np.where(np.isfinite(a), np.clip(np.round(a * 127), -127, 127), -128)
    return q.astype(np.int8)


def dec(v: float) -> str:
    """Plain decimal with 2 significant digits, never scientific (0.41,
    0.0014, 0.0000012)."""
    if v <= 0:
        return "0"
    import math
    d = max(0, 1 - int(math.floor(math.log10(v))))
    return f"{v:.{d}f}"


def top_tokens_html(tokenizer, ids, vals, total) -> str:
    """mean-ci-widget's top_token_label rules (top 8, cut below 0.5% share
    after the first 3), tokens bolded, HTML-escaped."""
    if total <= 0:
        return "(no CI in sample)"
    parts = []
    for tid, v in zip(ids, vals):
        share = float(v) / total
        if v <= 0 or len(parts) >= 8 or (len(parts) >= 3 and share < 0.005):
            break
        t = repr(tokenizer.decode([int(tid)]))
        if len(t) > 24:
            t = t[:21] + "…" + t[0]
        pct = f"{share:.1%}" if share < 0.095 else f"{share:.0%}"
        parts.append(f"<b>{html_mod.escape(t, quote=False)}</b> {pct}")
    return "top CI tokens: " + " · ".join(parts)


def old_labels():
    """mod -> {component id -> (label, source)} from mean-ci-widget's
    data.json (paper-site labels with interp.db fallback, all components)."""
    d = json.loads((MCW_CACHE / "data.json").read_text(encoding="utf-8"))
    lab = {}
    for subs in d["pile_4l"].values():
        for s in subs:
            lab[s["name"]] = {int(i): (l, sr) for i, l, sr in
                              zip(s["idx"], s["label"], s["src"])}
    return lab


def is_pos0(pos_fires, mod, cid) -> bool:
    fp = pos_fires[f"{mod}|Fp"][cid]
    tot = int(fp.sum())
    return tot >= 20 and fp[0] / tot > 0.5


def comp_meta(name, mod, cid, mean_ci, desc_html, pos_fires):
    s = (f"<b>{name} {mod}:{cid}</b> &nbsp;mean CI {dec(mean_ci)}<br>"
         + desc_html)
    if is_pos0(pos_fires, mod, cid):
        fp = pos_fires[f"{mod}|Fp"][cid]
        tot = int(fp.sum())
        s += (f"<br><b>pos-0 component</b> ({fp[0] / tot:.0%} of its {tot} "
              "CI&gt;0.1 fires are at position 0)")
    return s


def comp_signs(acts, mod):
    """Per-component sign gauge (+1/-1, full C axis): positive input
    activation on the majority of CI>0.1 tokens; ties broken by the summed
    firing activation; components that never fire by the all-token sum;
    still-ambiguous ones +1."""
    Npos, F = acts[f"{mod}|Npos"], acts[f"{mod}|F"]
    maj = np.where(2 * Npos > F, 1.0,
                   np.where(2 * Npos < F, -1.0, np.sign(acts[f"{mod}|Ssum"])))
    s = np.where(F > 0, maj, np.sign(acts[f"{mod}|Sall"]))
    s[s == 0] = 1.0
    return s


def match_perm(R, match_r):
    """Column permutation of the CI-ordered cross matrix R implementing the
    static reports' diagonal matching: matched columns (max r >= match_r) at
    their argmax row's position (ties by own mean CI = ascending column
    index), unmatched appended by mean CI. Returns (perm, n_matched)."""
    s = np.nan_to_num(R.astype(np.float32), nan=-1.0)
    best, smax = s.argmax(0), s.max(0)
    mi = np.flatnonzero(smax >= match_r)
    ui = np.flatnonzero(smax < match_r)
    mi = mi[np.argsort(best[mi], kind="stable")]
    return np.concatenate([mi, ui]).astype(int), int(len(mi))


def main() -> None:
    alive, stats, G, T = cross_stats()
    ids_ci, order_pos, ci_dump = ci_orders(alive)
    tokenizer = load_tokenizer("pile_4l")
    uv = {n: np.load(CACHE / f"uv_{n}.npz") for n in DEC}
    act_signs = {n: np.load(CACHE / f"act_signs_{n}.npz") for n in DEC}
    posf = {n: np.load(CACHE / f"pos_fires_{n}.npz") for n in DEC}
    tok = {n: np.load(MCW_CACHE / f"top_tokens_{n}.npz") for n in ("newA", "newB")}
    coci = {"old": np.load(CACHE / "coci_pile_4l.npz"),
            "newA": np.load(CACHE / "coci_newA.npz"),
            "newB": np.load(CACHE / "coci_newB.npz")}
    labels = old_labels()

    # per-decomposition, per-matrix: ids / hover meta / pos-0 flags, CI order
    dec_data = {n: {} for n in DEC}
    for name in DEC:
        for mod in MODS:
            ids = ids_ci[name][mod]
            ci = ci_dump[name][mod][order_pos[name][mod]]
            meta, p0 = [], []
            for k, (cid, mc) in enumerate(zip(ids, ci)):
                cid = int(cid)
                if name == "old":
                    lab, src = labels[mod][cid]
                    d_html = ("label: <i>"
                              + html_mod.escape(lab, quote=False) + "</i>")
                    if src != "paper-site label":
                        d_html += (' <span style="color:#888">('
                                   + html_mod.escape(src, quote=False)
                                   + ")</span>")
                else:
                    z = tok[name]
                    d_html = top_tokens_html(
                        tokenizer, z[f"{mod}|top_ids"][cid],
                        z[f"{mod}|top_ci"][cid], float(z[f"{mod}|total"][cid]))
                meta.append(comp_meta(name, mod, cid, float(mc), d_html,
                                      posf[name]))
                if is_pos0(posf[name], mod, cid):
                    p0.append(k)
            dec_data[name][mod] = {"ids": [str(int(c)) for c in ids],
                                   "meta": meta, "p0": p0}

    def signed_unit(name, mod, meas):
        """Sign-gauge-fixed unit factor rows in dump (ascending-alive) order:
        the same per-component sign flips U and V jointly."""
        sgn = comp_signs(act_signs[name], mod)[alive[f"{name}|{mod}"]]
        return unit(uv[name][f"{mod}|{meas}"]) * sgn[:, None].astype(np.float32)

    combos = {}
    blob_tags = {}
    total_blob = 0
    # same-decomposition combos: both axes CI order, symmetric -> triangle
    for name in DEC:
        combos[f"{name}|{name}"] = {}
        for mod in MODS:
            pos = order_pos[name][mod]           # CI order -> dump positions
            gids = ids_ci[name][mod]             # CI order -> global ids
            n = len(gids)
            R = coci[name][f"{mod}|r"].astype(np.float32)[np.ix_(gids, gids)]
            iu = np.triu_indices(n)
            parts = [q8(R[iu])]
            for meas in ("U", "V"):
                X = signed_unit(name, mod, meas)[pos]
                parts.append(q8((X @ X.T)[iu]))
            blob = zlib.compress(np.concatenate(parts).tobytes(), 9)
            total_blob += len(blob)
            blob_tags[f"{name}|{name}|{mod}"] = base64.b64encode(blob).decode()
            combos[f"{name}|{name}"][mod] = {}
        print(f"{name}|{name}: blobs done ({total_blob / 1e6:.0f} MB so far)",
              flush=True)

    # cross combos: rows = first decomposition's CI order, cols = second's;
    # the diagonal-matched orders for both display directions stored as
    # column permutations (fwd: y = first, x = second; rev: swapped)
    for P, Q in CROSS_PAIRS:
        combos[f"{P}|{Q}"] = {}
        for mod in MODS:
            R = pair_r(stats, G, T, P, Q, mod)[order_pos[P][mod]][
                :, order_pos[Q][mod]]
            parts = [q8(R)]
            for meas in ("U", "V"):
                Y = signed_unit(P, mod, meas)[order_pos[P][mod]]
                X = signed_unit(Q, mod, meas)[order_pos[Q][mod]]
                parts.append(q8(Y @ X.T))
            blob = zlib.compress(
                np.concatenate([p.ravel() for p in parts]).tobytes(), 9)
            total_blob += len(blob)
            perm_f, _ = match_perm(R, -np.inf)
            perm_ft, m_f = match_perm(R, MATCH_R)
            perm_r, _ = match_perm(R.T, -np.inf)
            perm_rt, m_r = match_perm(R.T, MATCH_R)
            blob_tags[f"{P}|{Q}|{mod}"] = base64.b64encode(blob).decode()
            combos[f"{P}|{Q}"][mod] = {
                "fwd": {"perm": perm_f.tolist(), "permT": perm_ft.tolist(),
                        "matched": m_f},
                "rev": {"perm": perm_r.tolist(), "permT": perm_rt.tolist(),
                        "matched": m_r},
            }
        print(f"{P}|{Q}: blobs done ({total_blob / 1e6:.0f} MB so far)",
              flush=True)

    data = {"decs": DEC, "mods": MODS, "dec": dec_data, "combo": combos}
    # base64 is [A-Za-z0-9+/=] only, so raw text inside a script tag is safe
    tags = "\n".join(f'<script type="text/plain" id="B|{k}">{b64}</script>'
                     for k, b64 in blob_tags.items())
    from plotly.offline import get_plotlyjs
    page = (PAGE
            .replace("__MATCH_R__", str(MATCH_R))
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__BLOBS__", tags)
            .replace("__DATA__",
                     json.dumps(data, separators=(",", ":"))
                     .replace("</", "<\\/")))
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>old / newA / newB — cross heatmaps (interactive)</title>
<script>__PLOTLYJS__</script>
<style>
  body { font-family: system-ui, sans-serif; margin: 14px 20px; color: #1a1a2e; }
  h2 { margin: 0 0 4px 0; font-size: 19px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 10px; max-width: 1100px; }
  .controls { display: flex; gap: 22px; align-items: center; margin-bottom: 6px;
              flex-wrap: wrap; font-size: 14px; }
  select { font-size: 14px; padding: 3px 6px; }
  #info { border: 1px solid #ccc; background: #fafafa; padding: 8px 12px;
          font-size: 13px; min-height: 96px; max-width: 1100px; margin-top: 6px;
          margin-bottom: 220px; line-height: 1.45; }
  #info .vals { color: #444; font-family: ui-monospace, monospace; }
  #plot { height: 575px; max-width: 1200px; }
</style>
</head>
<body>
<h2>old / newA / newB — cross heatmaps, all matrices</h2>
<div class="sub">Pick a decomposition for each axis (<b>old</b> =
s-55ea3f9b, the paper's decomposition; <b>newA</b> = p-8383f5e5, <b>newB</b> =
p-4d9a6a12). When they differ: y = descending mean CI, and each x component
is placed at the position of its best co-CI match (argmax r over the y
components, ties by own mean CI). With the <b>match threshold</b> on (the
static reports' ordering), components with max r &lt; __MATCH_R__ go to a
mean-CI-ordered right tail instead; with it off, they sit on the diagonal
band too, at their (possibly noisy) argmax row. When y and x show the
<b>same</b> decomposition, both axes are simply sorted by descending mean CI
(the threshold checkbox does not apply). Zoom by dragging a rectangle,
double-click to reset, shift-drag to pan. Hover a cell for both components'
details below the plot (newA/newB: top activating tokens; old: autointerp
label). cos(U)/cos(V) are <b>signed</b>: each component's (U, V) sign gauge
is fixed so that its input activation V&middot;x is positive on the majority
of tokens where the component is causally important (CI &gt; 0.1, over the
same 4,000 Pile rows).
<b style="color:#00a300">Green edge ticks</b> mark <b>pos-0
components</b> (&gt; 50% of their CI&gt;0.1 fires at chunk position 0): left
edge = y rows, top edge = x columns; they follow the cells when zooming.
Values are stored quantized to steps of 1/127 (&pm;0.004).</div>
<div class="controls">
  <label>y: <select id="ydec"></select></label>
  <label>x: <select id="xdec"></select></label>
  <label>Matrix:
    <select id="mat"></select>
  </label>
  <label><input type="radio" name="meas" value="r" checked> co-CI r</label>
  <label><input type="radio" name="meas" value="cu"> cos U (write)</label>
  <label><input type="radio" name="meas" value="cv"> cos V (read-in)</label>
  <label><input type="checkbox" id="thresh" checked> match threshold
    (r ≥ __MATCH_R__; unmatched at right)</label>
  <span id="matched" style="color:#666"></span>
</div>
<div id="plot"></div>
<div id="info">Hover a cell&hellip;</div>
<script id="D" type="application/json">__DATA__</script>
__BLOBS__
<script>
const DATA = JSON.parse(document.getElementById("D").textContent);
// matplotlib-style diverging scale with a PURE WHITE center (plotly's
// built-in RdBu midpoint is gray)
const RDBU = [[0, "rgb(5,48,97)"], [0.125, "rgb(33,102,172)"],
  [0.25, "rgb(67,147,195)"], [0.375, "rgb(146,197,222)"],
  [0.44, "rgb(209,229,240)"], [0.5, "rgb(255,255,255)"],
  [0.56, "rgb(253,219,199)"], [0.625, "rgb(244,165,130)"],
  [0.75, "rgb(214,96,77)"], [0.875, "rgb(178,24,43)"], [1, "rgb(103,0,31)"]];
const MEAS = {
  r:  {title: "co-CI r(CI)", colorscale: RDBU, zmin: -1, zmax: 1},
  cu: {title: "cos(U) (write vectors)", colorscale: RDBU, zmin: -1, zmax: 1},
  cv: {title: "cos(V) (read-in vectors)", colorscale: RDBU, zmin: -1, zmax: 1},
};
const matSel = document.getElementById("mat");
for (const mod of DATA.mods) {
  const o = document.createElement("option");
  o.value = o.textContent = mod;
  matSel.appendChild(o);
}
for (const id of ["ydec", "xdec"]) {
  const sel = document.getElementById(id);
  for (const n of DATA.decs) {
    const o = document.createElement("option");
    o.value = o.textContent = n;
    sel.appendChild(o);
  }
}
document.getElementById("ydec").value = "newA";
document.getElementById("xdec").value = "newB";
const measure = () =>
  document.querySelector('input[name="meas"]:checked').value;
const fmt = v => (v === null || v === undefined) ? "undefined" : v.toFixed(2);
const plot = document.getElementById("plot");
let hooked = false;
let cur = null;    // state of the currently rendered view (for hover)
const bufs = {};   // "combo|mod" -> Promise of {r, cu, cv: Int8Array}

// canonical stored pair: DATA.decs order; trans = y is the second-named
function comboOf(Y, X) {
  if (Y === X) return {key: Y + "|" + Y, same: true, trans: false};
  const [P, Q] = DATA.decs.filter(n => n === Y || n === X);
  return {key: P + "|" + Q, same: false, trans: Y === Q, P: P, Q: Q};
}

function getBufs(key, mod, nCells) {
  // caches the PROMISE so concurrent renders of the same view share one
  // decode; the base64 text lives in an inert script tag, pulled from the
  // DOM only here (keeping it out of the startup JSON.parse)
  const bk = key + "|" + mod;
  if (!bufs[bk]) bufs[bk] = (async () => {
    const bin = atob(document.getElementById("B|" + bk).textContent.trim());
    const raw = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) raw[i] = bin.charCodeAt(i);
    const out = new Uint8Array(await new Response(
      new Blob([raw]).stream().pipeThrough(
        new DecompressionStream("deflate"))).arrayBuffer());
    return {
      r: new Int8Array(out.buffer, 0, nCells),
      cu: new Int8Array(out.buffer, nCells, nCells),
      cv: new Int8Array(out.buffer, 2 * nCells, nCells),
    };
  })();
  return bufs[bk];
}
const q2v = q => q === -128 ? null : q / 127;

// value lookup in CI-axis coordinates (iy on the y decomposition's CI axis,
// ix on the x decomposition's) for the current view `c`
function ciVal(c, b, keyM, iy, ix) {
  if (c.same) {
    let i = iy, j = ix;
    if (i > j) { const t = i; i = j; j = t; }
    return q2v(b[keyM][i * (2 * c.n - i + 1) / 2 + (j - i)]);
  }
  return q2v(c.trans ? b[keyM][ix * c.nQ + iy] : b[keyM][iy * c.nQ + ix]);
}

async function render() {
  const Y = document.getElementById("ydec").value;
  const X = document.getElementById("xdec").value;
  const mod = matSel.value;
  const co = comboOf(Y, X);
  const dy = DATA.dec[Y][mod], dx = DATA.dec[X][mod];
  const ny = dy.ids.length, nx = dx.ids.length;
  const c = {same: co.same, trans: co.trans,
             n: ny, nQ: co.same ? ny : DATA.dec[co.Q][mod].ids.length};
  const nCells = co.same ? ny * (ny + 1) / 2
                         : DATA.dec[co.P][mod].ids.length * c.nQ;
  const b = await getBufs(co.key, mod, nCells);
  if (matSel.value !== mod || document.getElementById("ydec").value !== Y ||
      document.getElementById("xdec").value !== X) return;  // stale render
  const threshEl = document.getElementById("thresh");
  threshEl.disabled = co.same;
  const thr = threshEl.checked;
  // display column -> x decomposition's CI-axis index
  let perm, matched = null;
  if (co.same) {
    perm = dx.ids.map((_, j) => j);
  } else {
    const dir = DATA.combo[co.key][mod][co.trans ? "rev" : "fwd"];
    perm = thr ? dir.permT : dir.perm;
    if (thr) matched = dir.matched;
  }
  const m = MEAS[measure()], keyM = measure();
  const x = perm.map(k => dx.ids[k]);
  const z = new Array(ny);
  for (let i = 0; i < ny; i++) {
    const row = new Array(perm.length);
    for (let j = 0; j < perm.length; j++) row[j] = ciVal(c, b, keyM, i, perm[j]);
    z[i] = row;
  }
  document.getElementById("matched").textContent =
    ny + " × " + nx + " alive components" +
    (matched !== null ? " (" + matched + " matched)" : "");
  const trace = {
    type: "heatmap", z: z, x: x, y: dy.ids,
    colorscale: m.colorscale, zmin: m.zmin, zmax: m.zmax,
    hovertemplate: Y + " :%{y} × " + X + " :%{x}<br>" + m.title +
                   " = %{z:.2f}<extra></extra>",
  };
  // green edge ticks marking pos-0 components; data-coordinate positions so
  // they track the cells under zoom (columns mapped through the current
  // permutation)
  const disp = new Array(perm.length);
  perm.forEach((k, m_) => { disp[k] = m_; });
  const P0 = {color: "#00a300", width: 2};
  const shapes = dy.p0.map(i => ({
    type: "line", xref: "paper", yref: "y",
    x0: -0.012, x1: -0.002, y0: i, y1: i, line: P0,
  })).concat(dx.p0.map(k => ({
    type: "line", xref: "x", yref: "paper",
    x0: disp[k], x1: disp[k], y0: 1.003, y1: 1.015, line: P0,
  })));
  // uirevision keeps the user's zoom/pan across re-renders (measure, matrix
  // AND axis-decomposition switches; the kept range is index-based —
  // double-click resets)
  const layout = {
    title: {text: Y + " × " + X + "   " + mod + " — " + m.title,
            font: {size: 14}},
    margin: {l: 55, r: 20, t: 46, b: 45},
    uirevision: "keep",
    shapes: shapes,
    xaxis: {title: {text: X + " component id", font: {size: 12}},
            type: "category", showticklabels: nx <= 120},
    yaxis: {title: {text: Y + " component id", font: {size: 12}},
            type: "category", autorange: "reversed",
            showticklabels: ny <= 120, scaleanchor: "x"},
  };
  cur = {c: c, b: b, Y: Y, X: X, dy: dy, dx: dx, perm: perm};
  Plotly.react(plot, [trace], layout, {responsive: true});
  if (!hooked) {
    hooked = true;
    plot.on("plotly_hover", ev => {
      if (!cur) return;
      const p = ev.points[0];
      // ids are unique per axis, so the category labels map straight back to
      // CI-axis indices (the x permutation only affects display order)
      const i = cur.dy.ids.indexOf(p.y), j = cur.dx.ids.indexOf(p.x);
      if (i < 0 || j < 0) return;
      document.getElementById("info").innerHTML =
        '<div class="vals">co-CI r = ' + fmt(ciVal(cur.c, cur.b, "r", i, j)) +
        ' &nbsp;|&nbsp; cos U = ' + fmt(ciVal(cur.c, cur.b, "cu", i, j)) +
        ' &nbsp;|&nbsp; cos V = ' + fmt(ciVal(cur.c, cur.b, "cv", i, j)) +
        '</div>' +
        '<div>' + cur.dy.meta[i] + '</div><div style="margin-top:6px">' +
        cur.dx.meta[j] + '</div>';
    });
  }
}

matSel.addEventListener("change", render);
document.getElementById("ydec").addEventListener("change", render);
document.getElementById("xdec").addEventListener("change", render);
document.getElementById("thresh").addEventListener("change", render);
for (const r of document.querySelectorAll('input[name="meas"]'))
  r.addEventListener("change", render);
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
