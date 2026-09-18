# sink-harvest-widget

Interactive browsers for the three attention-sink decompositions **C / D / E**
(task-1423 harvest bundles published 2026-09-17; run↔name table in
[`../models_and_decomps.md`](../models_and_decomps.md)).

## `activation_examples.html` (+ sidecar data in `hide/data/` — move them together)

Per-component **activation examples browser**: select decomposition × matrix ×
component (alive components only, sorted by sample mean CI descending, with a
free-text filter and prev/next buttons) and see the component's top activating
windows — up to 16 non-overlapping 51-token windows of surrounding text, token
background depth (green) = causal importance, token underline = signed input
activation a = x·V (blue negative / white zero / red positive, scaled to the
component's max |a| over its shown windows; majority-positive sign gauge),
hover a token for its exact CI and activation. The header shows sample stats +
the co-authors'
10M-token harvest record for the component (harvest mean CI, firing count,
eligible regions, examples stored).

The example windows are **our own**, computed over the 4,000 cached Pile rows
(2.05M tokens) on Modal (`hide/examples_compute_modal.py` → 
`hide/cache/examples_{C,D,E}.npz` for CI traces, `hide/examples_act_modal.py`
→ `hide/cache/examples_act_{C,D,E}.npz` for activation traces at the same
windows, then `hide/build_examples.py` renders the page + sidecars). They are *not* the harvest's stored example windows — those
are unreadable until the co-authors share the non-public `HarvestReader`
(below). Same presentation conventions as the paper's viz app
(`prev_paper/param-decomp-vpd/param_decomp/app`, which reads only the old
harvest.db format). ⚠ CI computed through the public JAX loader
(broken-RoPE caveat, `../sink-models/rope_report.md`).

Known quirk: for dense always-on components (CI ≈ 1 everywhere, e.g.
h.0.attn.q_proj:282) the top-CI candidates are all ties, so the stored
windows cluster in the earliest cached rows and there may be only a couple of
them — examples of an always-on component are arbitrary anyway. Sparse
components (the interesting ones) get 16 clean windows.

## `harvest_widget.html` (27 MB, self-contained)

Dropdowns: decomposition (C/D/E) × matrix block (L0 Attn … L3 MLP) × view:

- **sorted curves** — per matrix, all components ranked by harvest mean CI
  (descending), with our own 2.05M-token sample mean CI drawn at the same
  ranks for comparison; dashed green line = the 1e-6 alive cutoff.
- **harvest vs sample** — log-log scatter of the two means per component,
  y = x reference, Spearman ρ in each panel title.

Hover shows the full per-component harvest record: firing count (lower-leaky
CI > 0.1, out of 10,002,432 corpus tokens), mean CI, mean / mean-abs / max-abs
activation, eligible region count, stored example count, plus the component's
top activating tokens from our sample.

## Data sources & caveats

- Harvest stats: `harvest.sqlite` in each bundle
  (`../sink-models/runs/<run>/harvest/<harvest-id>/`). Verified: the stored
  means equal the shard-parquet sums / 10,002,432.
- Sample mean CI + top tokens: `coci-heatmaps/hide/cache/coci_{C,D,E}.npz` and
  `mean-ci-widget/hide/cache/top_tokens_{C,D,E}.npz` — **⚠ computed through
  the public JAX loader whose RoPE is broken for the sink targets**
  (`../sink-models/rope_report.md`). The harvest presumably ran on the
  co-authors' internal (correct-RoPE?) code, so the scatter view doubles as a
  per-component probe of how much that mismatch matters. First look: Spearman
  ρ per matrix is 0.89–0.9998 (weakest in L2 attention and the MLPs), i.e.
  rank order mostly survives, but per-component disagreement is visible in
  the scatter tails.

## Not yet included: activation examples

Each harvest also stores up to 100 example windows (41 tokens, firing trace,
CI, normalized activation) per component in `examples.bin`, but the reader
code (`param_decomp.harvest.reader` from the bundle README) is **not in the
public repository** (checked all branches and history), and the binary format
is only partly reverse-engineered — see
[`examples_bin_format_notes.md`](examples_bin_format_notes.md). The blocking
issue is that the writer reuses per-component scratch buffers, so unused
bytes contain stale data from *other* components; without the exact section
layout, example windows risk being attributed to the wrong component. Ask the
co-authors for the reader (same request as the RoPE convention question)
before building an examples browser.

## Rebuild

- Stats widget: `python hide/build.py` (~30 s, default Python 3.11; needs the
  three extracted harvest bundles in `../sink-models/runs/` and the
  coci/top-tokens caches). Extracted data cached in `hide/cache/data.json` —
  delete to re-extract.
- Examples widget: `modal run --detach hide/examples_compute_modal.py`
  (~30 min A10G, three runs in parallel), `modal volume get vpd-4layer
  /examples_<X>.npz sink-harvest-widget/hide/cache/` for X = C, D, E, then
  `python hide/build_examples.py` (~2 min → the HTML + `hide/data/` sidecars).
