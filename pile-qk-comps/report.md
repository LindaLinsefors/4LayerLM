# pile-qk-comps — subcomponent weight per RoPE wavelength (pile_4l)

**Question:** for each decomposed q/k matrix, how much weight does each VPD
subcomponent place on each rotary wavelength?

## Definitions

RoPE acts inside attention by rotating query/key dim pairs. With head dim 128
and `rotary_adjacent_pairs: false`, plane $p$ pairs dims $(p,\ p+64)$ within
each head ($p = 0..63$) and rotates by angle $\theta_p(t) = t / 10000^{p/64}$
at position $t$, i.e. wavelength

$$\lambda_p = 2\pi \cdot 10000^{p/64} \ \text{tokens} \qquad (6.3 \to 54\text{k};\ \lambda_p > n_{ctx}=512 \text{ for } p \gtrsim 31).$$

Subcomponent $c$ of matrix $W$ writes the rank-one output
$U_c\,(V_c \cdot x)$ into q/k space (6 heads × 128, head-major). Its
rotation-invariant energy in plane $p$, summed over heads:

$$E_{c,p} = \sum_h \left( U_{c,\,128h+p}^2 + U_{c,\,128h+p+64}^2 \right),$$

shown as the per-component fraction $F_{c,p} = E_{c,p} / \sum_p E_{c,p}$
(independent of component magnitude — $\|V_c\|$ cancels; uniform = 1/64 ≈ 0.016).

## Files

- `qk_wavelengths.png` — one heatmap per matrix (4 layers × {q_proj, k_proj},
  each C = 512): **alive components only** (categories from
  `component-analasys`; user decision 2026-08-28), sorted by spectral center
  of mass, × 64 wavelength planes (log-λ axis = plane index). Figure rows are
  sized by each layer's alive counts. Dashed line: λ = n_ctx = 512.
  Color = $F_{c,p}$, capped at 0.06.
- `qk_heads.png` — same layout, but x = attention head instead of wavelength
  plane: $E_{c,h} = \sum_d U_{c,\,128h+d}^2$, row-normalized (uniform = 1/6),
  rows sorted by dominant head then descending dominance.
- `qk_heads_wavelengths.png` — the full 3-way view: x = 6 head blocks × 64
  planes (384 columns), $E_{c,h,p} = U_{c,\,128h+p}^2 + U_{c,\,128h+p+64}^2$
  row-normalized (uniform = 1/384), rows sorted by dominant head then spectral
  CoM. Red lines: head boundaries; dashed: λ = n_ctx within each block.
- `qk_wavelengths_mean.png` — mean $F_{c,p}$ over **alive** components only,
  per layer, q and k side by side.
- `qk_readin_cosine.png` — $|\cos(V_a, V_b)|$ between the read-in vectors of
  all 655 alive q/k components (blocks in layer/matrix order — black lines:
  layer divides, red lines: q|k divides; within each
  block, hierarchically clustered so similar directions sit together —
  average linkage on $1-|\cos|$ with optimal leaf ordering; color capped at
  0.3). Absolute value because the component sign is gauge:
  $(V_c, U_c) \to (-V_c, -U_c)$ leaves the rank-one component unchanged.
- `qk_readin_cosine_ci.png` — same matrix, but within each block components
  are sorted by descending mean causal importance (from harvest.db; cached in
  `hide/cache/mean_ci.npz`). The read-in clusters appear as scattered stripe
  patterns here — cluster membership is not a simple function of mean CI,
  though the densest similarity texture tends to sit at the high-CI end of
  each block (clearest in L2q).
- `hide/spectra.py` — compute + plot (cache: `hide/cache/energies.npz`).

## Observations (2026-08-28)

- **Layer 2 stands out sharply.** A large block of its alive components
  (~half of q's 92, and the bottom ~third of k's 167) concentrates energy in
  the slow planes (λ ≳ 1–2k ≫ n_ctx) while being strongly depleted at short
  wavelengths. Within the context window those planes rotate by ≪ 2π, so they
  act as smooth monotone functions of absolute position — these look like
  **coarse-position / long-range components** rather than oscillatory ones. In
  the layer-2 q mean spectrum the effect is big: short-λ planes carry ~0.005
  vs ~0.03 at long λ (6× ratio).
- **Layers 0, 1, 3** are much flatter; all matrices show a mild common tilt
  toward long wavelengths (mean fraction rises from ~0.010–0.013 at λ ≈ 6
  to ~0.018–0.02 above n_ctx). No alive component anywhere specializes in
  *fast* planes (no crisp short-wavelength block).
- **Alive counts are small**: q_proj 111 / 15 / 92 / 36 and k_proj
  126 / 48 / 167 / 60 for layers 0–3 (of C = 512 each) — layer 1 attention
  barely uses q/k subcomponents at all.
- **Head view (`qk_heads.png`): components are mostly cross-head.** Typical
  dominant-head fraction is only ~0.2–0.4 (uniform = 1/6), consistent with the
  paper's cross-head subcomponent finding. The main exception is layer 2,
  where sizeable single-head blocks exist — and **layer-2 head 4 is the
  position head**: of its head-4-dominant components (fraction > 0.4),
  25/29 (q) and 23/27 (k) are the same components that put > 75% of their
  energy into the slow λ ≳ 1.1k planes (23/28 of all such slow k components
  are head-4-dominant). So the coarse-position block and the head-4 block are
  one and the same mechanism, present on both the q and k side as VPD would
  require for it to affect attention scores.
- **3-way view (`qk_heads_wavelengths.png`)**: the long-λ tilt is *not*
  uniform over heads. In layer 0 (both q and k) nearly every alive component
  shares one dense vertical band — head 4 at λ ≳ 630 — so the layer-0 tilt is
  mostly a single shared head-4 slow-plane direction rather than per-component
  structure. In layer 2 the slow-plane energy of the position components is
  spread over heads 2–4 (densest in head 4) on both sides, while h.2.k's
  head-5-dominant group (~rows 92–167) is broadly spread over heads 0/1/5 and
  avoids heads 2–4 — two roughly disjoint head-sets. Layer 3's head-4 block
  again pairs with slow planes.
- **Read-in geometry (`qk_readin_cosine.png`): near-orthogonal overall.**
  Median off-diagonal $|\cos| = 0.026$, i.e. at the random-direction baseline
  $\mathbb{E}|\cos| = \sqrt{2/(\pi\cdot768)} = 0.029$; p99 = 0.13, max = 0.79,
  only 24 pairs above 0.5. Block means barely move off baseline — largest are
  within-L1 (L1q 0.065, L1k 0.069, L1q×L1k cross 0.043 — layer 1's few alive
  components share a small read-in subspace) and the later-layer diagonals
  (L2, L3 ≈ 0.04–0.05). After within-block clustering the excess resolves
  into a handful of tight clusters: a dense ~20-component cluster in L1k
  (echoed by a small aligned L1q group), one tight cluster each in L2k and
  L3k that are *mutually aligned across layers* (dark off-diagonal L2k×L3k
  blocks — the two layers' k components reading the same residual
  directions), a few small L2q clusters, and tight pairs/triplets in L0.
  The 24 strong pairs (|cos| > 0.5) sit almost entirely there: 9 within L2k,
  5 L2k×L3k, plus a few in L0 (incl. 2 q–k pairs) and L3. Notably the
  layer-2 q×k position blocks do *not* read the same directions (L2q×L2k
  mean 0.032 ≈ baseline) — matching q and k position components are paired
  by their *output* planes, not by a shared input direction.
- Semi/never components (not shown in the current alive-only figure; observed
  in an earlier all-component version) have noisy near-uniform spectra
  (consistent with `component-analasys`: dead components are near-copies with
  no structure), though layer 0's dead components share the same long-λ tilt
  as its alive ones.

## Side checks: per-token CI of individual components (2026-08-28)

The `ComponentModel` + CI-function machinery runs locally in the 3.13 venv
(`prev_paper\param-decomp-vpd\.venv\Scripts\python.exe`); scripts `hide/ci_q68.py`,
`hide/ci_q68_sweep.py`, `hide/ci_the.py`.

- **q:68 ("A:" answer marker)** — its real trigger is the StackExchange
  `\n\nA:\n\n` separator, not generic "A:" (zero CI on a row with 15 table
  "A:"s). CI is position-*independent*: embedding a real firing context at
  absolute positions 21–508 gives CI 0.87–0.99 everywhere. The apparent
  end-of-text concentration in the paper widget is an artifact of its
  16-token display window.
- **The four "the" components (q:38, 212, 270, 345)** — near-duplicates:
  activations on 'the' tokens correlate r = 0.96–1.00 pairwise, CI patterns
  are identical (r = 1.00), i.e. they co-fire as one mechanism split into four
  rank-one pieces. All four *activate* (|a| ≈ 3–5 vs ≈ 0.7 background) on
  **every** 'the' regardless of occurrence rank, but are causally important
  essentially only when 'the' sits at the **very start of the 512-token
  chunk**: in 32 rows (548 'the's) the only CI event was 'The' at position 1;
  harvest-wide their CI firings have median left-context 1 token, 86–100%
  within 20 tokens of the row start, and only 0–7% follow an EOS — so it is
  chunk-start (cold-start queries with no context), not first-in-document.

## Per-matrix component-group reports (2026-08-28, regrouped same day)

`L<l>/L<l>_<q|k>.md` (8 reports, figures in `hide/figures/L<l>-Attn-<q|k>/`):
every alive component of each q/k
matrix grouped by **expected per-token CI co-firing**, judged by Claude from
each component's **full website description** (label + autointerp reasoning,
fetched from static.goodfire.ai/vpd-blog-post into
`hide/cache/site_descriptions.json`; the local interp.db labels proved too
vague — they scattered e.g. the 26-component L0q hacker-news-hyphen group).
Same trigger tokens/positions → same group; subset-detectors merged;
different tokens kept apart (q: vs a:); bimodal/vague descriptions left
ungrouped (198 of 655). Per group with >1 member, one figure with three
annotated pairwise grids over a 2.05M-token Pile sample (4000 cached rows,
computed on Modal — `hide/groups_compute_modal.py`, stats in
`hide/cache/groups_stats.npz`; text/figures generated by
`hide/groups_report.py` from `hide/groups_def.py`), plus, per report, an
`all_pairs.png` (all alive components × all three measures, ordered by group)
and a large single-panel `co_ci_full.png` (the co-CI grid alone with every
component index labeled on both axes, same order).

Alternative data-driven version for L0 k (2026-08-29): `L0/L0_k_coci.md`
(figures in `hide/figures/L0-Attn-k-coci/`, generated by
`hide/groups_coci_report.py` —
works for any of the 8 matrices) groups the same components by the measured
co-CI itself instead of descriptions: average-linkage clustering on
$1 - r(\mathrm{CI})$, cut so within-cluster mean $r \ge 0.5$; singletons →
ungrouped. Result: 21 clusters + 41 ungrouped (of 126); the big
description-based groups reappear (cluster 1 = the 17-strong q: machinery)
plus finer data-only pairs/triples the descriptions didn't predict.

## Position-in-chunk dependence of L0/L1 CI (2026-08-29)

`L<l>/L<l>_position_dependence.md` (L0, L1): for every alive q/k component, the
distribution of its CI firings (CI > 0.1) over the 512 chunk positions, from
per-position CI stats over the 4,000 cached rows (Modal:
`hide/pos_ci_compute_modal.py` → `hide/cache/pos_ci.npz`, all 8 q/k matrices
cached; report: `hide/pos_ci_report.py <layer>`). Chunk cuts are random within
documents, so the token distribution is position-independent and any profile
structure is genuine position dependence. Metric: excess total-variation from
uniform over 64 position bins (Monte-Carlo noise-corrected). The same metric
is computed for **activation firings** ($|a_c| > 1$) as a control: for L0 the
module input is a function of the token id alone, so its activation profiles
are position-flat by architecture — and indeed max a-excess TV = 0.00 there;
notably L1 is *also* flat (max 0.03), so even where the input carries
positional information the position-locking lives entirely in the CI
function, not in the activation magnitudes. Also recorded: q components
never have CI at position 0 in any layer (softmax over the single visible
key is constant ⇒ the query has no causal effect), while k CI is *largest*
at position 0 (key 0 feeds every later query — the sink site). Findings: in
both layers the split is clean — every flagged component is
chunk-start-concentrated, everything else is flat (excess TV ≈ 0).
**L0**: 5 of 111 q (the four 'the' components 38/212/270/345 + q:46
newlines, median position 4) and 12 of 126 k (number triple 14/31/413, 'of'
cluster 9/141/266, articles 353/498, determiners 273, open-parens 257 —
median positions 0–8; 201/243 only early-half-biased). **L1**: even more
extreme — 7 of its mere 15 alive q (excess TV 0.55–0.93, medians 1–2,
99–100% of fires at position < 8: 37/149/268/271/342/474/497 — most of
layer-1 q attention is chunk-start machinery) and 10 of 48 k (315/339/272/
121/147 literally labeled first-token/early-sequence; + 52/196/318/357;
k:327 "sequence boundaries and document starts" is 67% chunk-start with the
rest presumably at the uniformly-distributed EOS positions). Firing counts
of the L1 start components are large (up to 9k = 2.3 per chunk), i.e. they
fire on the first ~1–3 tokens of essentially every chunk regardless of
token identity — cold-start attention machinery, consistent with the
resid-stage finding that position 0 gets sink treatment from layer 1 on.
Validation-by-eye: the L0q hn-hyphen group is a solid r≈1 co-CI block in the
all-pairs figure, with near-zero read-in cosines throughout.

- **co-activation**: Pearson $r$ of $|a_c| = |\|U_c\|(V_c\cdot\varphi)|$ per token,
- **co-CI**: Pearson $r$ of the causal importance (lower_leaky) per token,
- **input cosine**: $|\cos(V_a, V_b)|$.

Headline pattern (e.g. L0k end-of-text group): same-function components
typically co-fire strongly in CI (r up to ~0.95) while their read-in vectors
stay near-orthogonal and co-activation is modest (~0.2–0.4) — shared *when*,
different *what*. The four L0q "the" components (co-activation ≈ 1.0) are the
exception, not the rule.

## Gotchas

- Wavelength is a property of the *plane index*, shared across heads — the
  energy here is summed over heads; head-resolved structure is not shown.
- The λ > n_ctx planes never complete a cycle in a 512-token context (the
  slowest rotate by only ~3–5° over the whole window); energy there encodes
  near-linear position readout, not periodicity.
- $F$ measures *output-side* weight geometry only — it says which planes a
  component *can* write to, not how the attention scores use them (that also
  depends on the matching q/k component on the other side).
