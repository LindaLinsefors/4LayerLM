"""Build the compare-decomps widget: interactive co-CI / cos(U) / cos(V)
heatmaps for ANY pair of the six decompositions

  old  = s-55ea3f9b  (target t-9d2b8f02,  the paper's run)
  newA = p-8383f5e5  (target t-9d2b8f02,  800k steps)
  newB = p-4d9a6a12  (target t-9d2b8f02,  800k steps, 10x weaker minimality)
  C    = p-d60af588  (target t-87f91319,  attention-sink model, seed 0)
  D    = p-fecd6a6b  (target t-87f91319,  attention-sink model, seed 1)
  E    = p-bd411e35  (target t-75f6c439,  attention-sink model)

Same UI as coci-heatmaps' interactive_newA_newB.html (y/x decomposition
dropdowns, matrix dropdown, measure radio, match-threshold checkbox, pos-0
green ticks, hover info panel), with one rule added: cos(U)/cos(V) are only
offered when both axes decompose the SAME target model (pairs within
{old, newA, newB}, the pair {C, D}, and every same-decomposition view) —
across different models the factors live in unrelated bases, so only co-CI
(same 2.05M training tokens for all six) is shown.

SIDECAR LAYOUT (user decision 2026-09-08): unlike the single-file coci
widget, the heatmap blobs live in hide/data/<a>--<b>--<mod>.js (one per
combo x matrix, ~300 files, loaded on demand via <script src> — works from
file://), so ../compare_decomps.html itself stays ~20 MB and RAM only grows
with the views actually opened. The HTML and hide/ must move together.

Inputs: coci-heatmaps/hide/cache/* for old/newA/newB (coci_*, mean_ci_pile_4l,
cross_alive, cross_partial_*, uv_*, act_signs_*, pos_fires_*),
mean-ci-widget/hide/cache/* (data.json labels, top_tokens_*), and
hide/cache/* here for C/D/E (from sink_stats_modal.py + cross_all_modal.py +
sink-models' pos_fires job).

Usage: python compare-decomps/hide/build.py   (~15-25 min)
"""

import base64
import html as html_mod
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CACHE = HERE / "cache"
CH = ROOT / "coci-heatmaps" / "hide" / "cache"
MCW = ROOT / "mean-ci-widget" / "hide" / "cache"
OUT_HTML = HERE.parent / "compare_decomps.html"
DATA_DIR = HERE / "data"

sys.path.insert(0, str(ROOT))
from load import load_tokenizer  # noqa: E402

DEC = ["old", "newA", "newB", "C", "D", "E"]
TARGET = {"old": "t-9d2b8f02", "newA": "t-9d2b8f02", "newB": "t-9d2b8f02",
          "C": "t-87f91319", "D": "t-87f91319", "E": "t-75f6c439"}
RUN_ID = {"old": "s-55ea3f9b", "newA": "p-8383f5e5", "newB": "p-4d9a6a12",
          "C": "p-d60af588", "D": "p-fecd6a6b", "E": "p-bd411e35"}
TRIO = {"old", "newA", "newB"}
MODS = [f"h.{l}.{m}" for l in range(4)
        for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                  "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]
MATCH_R = 0.3


# ---------- small helpers (shared with coci-heatmaps/hide/interactive_cross) --

def unit(rows):
    x = rows.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = np.inf
    return x / n


def q8(a):
    """float array in [-1, 1] -> int8 (value*127), NaN -> -128."""
    q = np.where(np.isfinite(a), np.clip(np.round(a * 127), -127, 127), -128)
    return q.astype(np.int8)


def dec_fmt(v: float) -> str:
    if v <= 0:
        return "0"
    import math
    d = max(0, 1 - int(math.floor(math.log10(v))))
    return f"{v:.{d}f}"


def top_tokens_html(tokenizer, ids, vals, total) -> str:
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
    d = json.loads((MCW / "data.json").read_text(encoding="utf-8"))
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


def comp_signs(acts, mod):
    """Sign gauge: positive input activation on the majority of CI>0.1 tokens
    (ties by summed firing activation, never-firing by the all-token sum)."""
    Npos, F = acts[f"{mod}|Npos"], acts[f"{mod}|F"]
    maj = np.where(2 * Npos > F, 1.0,
                   np.where(2 * Npos < F, -1.0, np.sign(acts[f"{mod}|Ssum"])))
    s = np.where(F > 0, maj, np.sign(acts[f"{mod}|Sall"]))
    s[s == 0] = 1.0
    return s


def match_perm(R, match_r):
    """Diagonal-matching column permutation of a CI-ordered cross matrix."""
    s = np.nan_to_num(R.astype(np.float32), nan=-1.0)
    best, smax = s.argmax(0), s.max(0)
    mi = np.flatnonzero(smax >= match_r)
    ui = np.flatnonzero(smax < match_r)
    mi = mi[np.argsort(best[mi], kind="stable")]
    return np.concatenate([mi, ui]).astype(int), int(len(mi))


# ---------------------------- data plumbing ----------------------------------

class Decomp:
    """Per-decomposition caches + orderings."""

    def __init__(self, name):
        self.name = name
        my = name not in TRIO
        self.coci = np.load((CACHE if my else CH) /
                            (f"coci_{name}.npz" if name != "old"
                             else "coci_pile_4l.npz"))
        if name == "old":
            self.mean = {m: np.load(CH / "mean_ci_pile_4l.npz")[m]
                         for m in MODS}
        else:
            self.mean = {m: self.coci[f"{m}|mean"] for m in MODS}
        if my:
            self.alive = {m: np.flatnonzero(self.mean[m] > 1e-6) for m in MODS}
        else:
            z = np.load(CH / "cross_alive.npz")
            self.alive = {m: z[f"{name}|{m}"] for m in MODS}
        self.uv = np.load((CACHE if my else CH) / f"uv_{name}.npz")
        self.uv_all_rows = my  # C/D/E dumps hold ALL components' rows
        self.acts = np.load((CACHE if my else CH) / f"act_signs_{name}.npz")
        self.posf = np.load(CACHE / f"pos_fires_sink_{name}.npz" if my
                            else CH / f"pos_fires_{name}.npz")
        # CI order (desc mean, ties by id) as global ids + dump positions
        self.ids_ci, self.order_pos, self.ci_sorted = {}, {}, {}
        for m in MODS:
            a = self.alive[m]
            ma = self.mean[m][a]
            pos = np.argsort(-ma, kind="stable")
            self.ids_ci[m] = a[pos]
            self.order_pos[m] = pos
            self.ci_sorted[m] = ma[pos]

    def signed_unit(self, mod, meas):
        """Sign-gauge-fixed unit factors, rows in dump (ascending-alive)
        order."""
        rows = self.uv[f"{mod}|{meas}"]
        if self.uv_all_rows:
            rows = rows[self.alive[mod]]
        sgn = comp_signs(self.acts, mod)[self.alive[mod]]
        return unit(rows) * sgn[:, None].astype(np.float32)

    def same_r(self, mod):
        """Alive x alive co-CI r in CI order (symmetric)."""
        gids = self.ids_ci[mod]
        return self.coci[f"{mod}|r"].astype(np.float32)[np.ix_(gids, gids)]


def trio_pair_r(mod, a, b):
    """Cross r for the pile-trio pairs from the old cross_partial caches,
    dump-axis order (the report_cross.py pair_r logic, self-contained)."""
    pA = np.load(CH / "cross_partial_newA.npz")
    pB = np.load(CH / "cross_partial_newB.npz")
    T = float(pA["T"])
    part = {("old", "newA"): pA, ("old", "newB"): pB, ("newA", "newB"): pB}[(a, b)]
    stats_src = {"old": pA, "newA": pA, "newB": pB}

    def mu_sd(n):
        s1, s2 = stats_src[n][f"{mod}|S1_{n}"], stats_src[n][f"{mod}|S2_{n}"]
        mu = s1 / T
        return mu, np.sqrt(np.clip(s2 / T - mu**2, 0, None))

    mua, sda = mu_sd(a)
    mub, sdb = mu_sd(b)
    g = part[f"{mod}|G_{a}_{b}"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (g / T - np.outer(mua, mub)) / np.outer(sda, sdb)
    r[~np.isfinite(r)] = np.nan
    return r


def main() -> None:
    tokenizer = load_tokenizer("pile_4l")
    labels = old_labels()
    d = {n: Decomp(n) for n in DEC}
    cross12 = np.load(CACHE / "cross_r_new12.npz")
    tok_src = {n: np.load((CACHE if n in ("C", "D", "E") else MCW) /
                          f"top_tokens_{n}.npz")
               for n in DEC if n != "old"}

    # per-decomposition hover meta / p0, CI order
    dec_data = {n: {} for n in DEC}
    for n in DEC:
        for mod in MODS:
            # one decompression per matrix — an NpzFile [key] access
            # decompresses the whole array every time
            fp_all = d[n].posf[f"{mod}|Fp"]
            tok_ids = tok_ci = tok_tot = None
            if n != "old":
                z = tok_src[n]
                tok_ids, tok_ci = z[f"{mod}|top_ids"], z[f"{mod}|top_ci"]
                tok_tot = z[f"{mod}|total"]
            meta, p0 = [], []
            for k, (cid, mc) in enumerate(zip(d[n].ids_ci[mod],
                                              d[n].ci_sorted[mod])):
                cid = int(cid)
                if n == "old":
                    lab, src = labels[mod][cid]
                    dh = ("label: <i>"
                          + html_mod.escape(lab, quote=False) + "</i>")
                    if src != "paper-site label":
                        dh += (' <span style="color:#888">('
                               + html_mod.escape(src, quote=False) + ")</span>")
                else:
                    dh = top_tokens_html(tokenizer, tok_ids[cid], tok_ci[cid],
                                         float(tok_tot[cid]))
                s = (f"<b>{n} {mod}:{cid}</b> &nbsp;mean CI {dec_fmt(float(mc))}"
                     f"<br>{dh}")
                fp = fp_all[cid]
                tot = int(fp.sum())
                if tot >= 20 and fp[0] / tot > 0.5:
                    s += (f"<br><b>pos-0 component</b> ({fp[0] / tot:.0%} of "
                          f"its {tot} CI&gt;0.1 fires are at position 0)")
                    p0.append(k)
                meta.append(s)
            dec_data[n][mod] = {"ids": [str(int(c)) for c in d[n].ids_ci[mod]],
                                "meta": meta, "p0": p0}
        print(f"meta {n} done", flush=True)

    DATA_DIR.mkdir(exist_ok=True)
    for f in DATA_DIR.glob("*.js"):
        f.unlink()
    combos = {}
    has_cos = {}
    total_blob = 0
    for i, a in enumerate(DEC):
        for b in DEC[i:]:
            key = f"{a}|{b}"
            has_cos[key] = TARGET[a] == TARGET[b]
            combos[key] = {}
            for mod in MODS:
                if a == b:
                    R = d[a].same_r(mod)
                    n = R.shape[0]
                    iu = np.triu_indices(n)
                    parts = [q8(R[iu])]
                    if has_cos[key]:
                        for meas in ("U", "V"):
                            X = d[a].signed_unit(mod, meas)[d[a].order_pos[mod]]
                            parts.append(q8((X @ X.T)[iu]))
                    combos[key][mod] = {}
                else:
                    if a in TRIO and b in TRIO:
                        r_dump = trio_pair_r(mod, a, b)
                    else:
                        r_dump = cross12[f"{a}|{b}|{mod}|r"].astype(np.float32)
                    R = r_dump[d[a].order_pos[mod]][:, d[b].order_pos[mod]]
                    parts = [q8(R)]
                    if has_cos[key]:
                        for meas in ("U", "V"):
                            Y = d[a].signed_unit(mod, meas)[d[a].order_pos[mod]]
                            X = d[b].signed_unit(mod, meas)[d[b].order_pos[mod]]
                            parts.append(q8(Y @ X.T))
                    pf, _ = match_perm(R, -np.inf)
                    pft, mf = match_perm(R, MATCH_R)
                    pr, _ = match_perm(R.T, -np.inf)
                    prt, mr = match_perm(R.T, MATCH_R)
                    combos[key][mod] = {
                        "fwd": {"perm": pf.tolist(), "permT": pft.tolist(),
                                "matched": mf},
                        "rev": {"perm": pr.tolist(), "permT": prt.tolist(),
                                "matched": mr}}
                blob = zlib.compress(
                    np.concatenate([p.ravel() for p in parts]).tobytes(), 9)
                total_blob += len(blob)
                b64 = base64.b64encode(blob).decode()
                (DATA_DIR / f"{a}--{b}--{mod}.js").write_text(
                    f'__reg("{key}|{mod}","{b64}");', encoding="ascii")
            print(f"{key}: done ({total_blob / 1e6:.0f} MB so far)", flush=True)

    data = {"decs": DEC, "mods": MODS, "runid": RUN_ID, "target": TARGET,
            "cos": has_cos, "dec": dec_data, "combo": combos}
    from plotly.offline import get_plotlyjs
    page = (PAGE
            .replace("__MATCH_R__", str(MATCH_R))
            .replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__DATA__",
                     json.dumps(data, separators=(",", ":"))
                     .replace("</", "<\\/")))
    OUT_HTML.write_text(page, encoding="utf-8")
    n_files = len(list(DATA_DIR.glob("*.js")))
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB) + "
          f"{n_files} data files ({total_blob / 1e6:.0f} MB compressed)")


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>decomposition comparison — old/newA/newB/C/D/E</title>
<script>__PLOTLYJS__</script>
<style>
  body { font-family: system-ui, sans-serif; margin: 14px 20px; color: #1a1a2e; }
  h2 { margin: 0 0 4px 0; font-size: 19px; }
  .sub { color: #666; font-size: 13px; margin-bottom: 10px; max-width: 1100px; }
  .controls { display: flex; gap: 22px; align-items: center; margin-bottom: 6px;
              flex-wrap: wrap; font-size: 14px; }
  select { font-size: 14px; padding: 3px 6px; }
  label.off { color: #aaa; }
  #info { border: 1px solid #ccc; background: #fafafa; padding: 8px 12px;
          font-size: 13px; min-height: 96px; max-width: 1100px; margin-top: 6px;
          margin-bottom: 220px; line-height: 1.45; }
  #info .vals { color: #444; font-family: ui-monospace, monospace; }
  #plot { height: 575px; max-width: 1200px; }
</style>
</head>
<body>
<h2>Decomposition comparison — old / newA / newB / C / D / E</h2>
<div class="sub">Pick a decomposition for each axis: <b>old</b> = s-55ea3f9b,
<b>newA</b> = p-8383f5e5, <b>newB</b> = p-4d9a6a12 (all of pile_4l target
t-9d2b8f02); <b>C</b> = p-d60af588 and <b>D</b> = p-fecd6a6b (attention-sink
model t-87f91319, seeds 0/1); <b>E</b> = p-bd411e35 (attention-sink model
t-75f6c439). <b>co-CI r</b> is available for every pair (all six are
evaluated on the same 2.05M Pile tokens); <b>cos(U)/cos(V)</b> only when both
axes decompose the SAME target model — across different models the factors
live in unrelated bases. Cosines are signed (each component's (U, V) gauge
fixed so its input activation V&middot;x is positive on the majority of its
CI&gt;0.1 tokens). Axes: y = descending mean CI; when the decompositions
differ each x component sits at its best co-CI match's row (with the
<b>match threshold</b> on, components with max r &lt; __MATCH_R__ go to a
mean-CI-ordered right tail); same decomposition on both axes = both sorted by
mean CI. Zoom by dragging, double-click to reset, shift-drag to pan; hover a
cell for both components' details (newA/newB/C/D/E: top activating tokens;
old: autointerp label). <b style="color:#00a300">Green edge ticks</b> mark
pos-0 components (&gt; 50% of CI&gt;0.1 fires at chunk position 0). Values
quantized to steps of 1/127. Heatmap data loads on demand from
<code>hide/data/</code> — keep that folder next to this file.</div>
<div class="controls">
  <label>y: <select id="ydec"></select></label>
  <label>x: <select id="xdec"></select></label>
  <label>Matrix:
    <select id="mat"></select>
  </label>
  <label id="l_r"><input type="radio" name="meas" value="r" checked> co-CI r</label>
  <label id="l_cu"><input type="radio" name="meas" value="cu"> cos U (write)</label>
  <label id="l_cv"><input type="radio" name="meas" value="cv"> cos V (read-in)</label>
  <label><input type="checkbox" id="thresh" checked> match threshold
    (r ≥ __MATCH_R__; unmatched at right)</label>
  <span id="matched" style="color:#666"></span>
</div>
<div id="plot"></div>
<div id="info">Hover a cell&hellip;</div>
<script id="D" type="application/json">__DATA__</script>
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
const bufs = {};   // "combo|mod" -> Promise of {r, cu?, cv?: Int8Array}
const pending = {};

// canonical stored pair: DATA.decs order; trans = y is the second-named
function comboOf(Y, X) {
  if (Y === X) return {key: Y + "|" + Y, same: true, trans: false};
  const [P, Q] = DATA.decs.filter(n => n === Y || n === X);
  return {key: P + "|" + Q, same: false, trans: Y === Q, P: P, Q: Q};
}

window.__reg = (bk, b64) => {
  const p = pending[bk];
  if (!p) return;
  delete pending[bk];
  (async () => {
    try {
      const bin = atob(b64);
      const raw = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) raw[i] = bin.charCodeAt(i);
      const out = new Uint8Array(await new Response(
        new Blob([raw]).stream().pipeThrough(
          new DecompressionStream("deflate"))).arrayBuffer());
      const n = p.nCells;
      const o = {r: new Int8Array(out.buffer, 0, n)};
      if (p.nMeas === 3) {
        o.cu = new Int8Array(out.buffer, n, n);
        o.cv = new Int8Array(out.buffer, 2 * n, n);
      }
      p.res(o);
    } catch (e) { p.rej(e); }
  })();
};

function getBufs(key, mod, nCells, nMeas) {
  // one <script src> fetch + decode per view, shared via a cached promise
  const bk = key + "|" + mod;
  if (!bufs[bk]) bufs[bk] = new Promise((res, rej) => {
    pending[bk] = {res: res, rej: rej, nCells: nCells, nMeas: nMeas};
    const s = document.createElement("script");
    s.src = "hide/data/" + key.replace("|", "--") + "--" + mod + ".js";
    s.onerror = () => { delete pending[bk];
                        rej(new Error("cannot load " + s.src)); };
    document.head.appendChild(s);
  });
  return bufs[bk];
}
const q2v = q => q === -128 ? null : q / 127;

// value lookup in CI-axis coordinates for the current view `c`
function ciVal(c, b, keyM, iy, ix) {
  if (!b[keyM]) return null;
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
  const cosOK = DATA.cos[co.key];
  // gray out / recover the cosine radios per combo
  for (const v of ["cu", "cv"]) {
    const inp = document.querySelector('input[name="meas"][value="' + v + '"]');
    inp.disabled = !cosOK;
    document.getElementById("l_" + v).className = cosOK ? "" : "off";
    document.getElementById("l_" + v).title =
      cosOK ? "" : "different target models — cosines not comparable";
  }
  if (!cosOK && measure() !== "r")
    document.querySelector('input[name="meas"][value="r"]').checked = true;
  const dy = DATA.dec[Y][mod], dx = DATA.dec[X][mod];
  const ny = dy.ids.length, nx = dx.ids.length;
  const c = {same: co.same, trans: co.trans,
             n: ny, nQ: co.same ? ny : DATA.dec[co.Q][mod].ids.length};
  const nCells = co.same ? ny * (ny + 1) / 2
                         : DATA.dec[co.P][mod].ids.length * c.nQ;
  const b = await getBufs(co.key, mod, nCells, cosOK ? 3 : 1);
  if (matSel.value !== mod || document.getElementById("ydec").value !== Y ||
      document.getElementById("xdec").value !== X) return;  // stale render
  const threshEl = document.getElementById("thresh");
  threshEl.disabled = co.same;
  const thr = threshEl.checked;
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
      const i = cur.dy.ids.indexOf(p.y), j = cur.dx.ids.indexOf(p.x);
      if (i < 0 || j < 0) return;
      const cu = ciVal(cur.c, cur.b, "cu", i, j);
      const cv = ciVal(cur.c, cur.b, "cv", i, j);
      const cosTxt = cur.b.cu
        ? ' &nbsp;|&nbsp; cos U = ' + fmt(cu) + ' &nbsp;|&nbsp; cos V = ' + fmt(cv)
        : ' &nbsp;|&nbsp; <span style="color:#888">cos U/V n/a (different ' +
          'target models)</span>';
      document.getElementById("info").innerHTML =
        '<div class="vals">co-CI r = ' + fmt(ciVal(cur.c, cur.b, "r", i, j)) +
        cosTxt + '</div>' +
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
