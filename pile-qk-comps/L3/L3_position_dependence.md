# L3 — q/k components with position-in-chunk-dependent causal importance

Per alive component of `h.3.attn.q_proj` / `h.3.attn.k_proj`: where in the 512-token chunk do its CI firings (CI > 0.1) happen? Over the 4,000 cached Pile rows every chunk position is seen exactly 4,000 times, and chunk boundaries fall at random points inside documents, so the token distribution is the same at every position (position 0 is *not* usually a document start — 74.6% of chunks contain no EOS). A component whose CI depends only on token/context therefore has a flat firing-position profile; any structure is genuine position dependence. Computed on Modal (`hide/pos_ci_compute_modal.py` → `hide/cache/pos_ci.npz`); tables/figures from `hide/pos_ci_report.py`.

**Metric:** profile $w_b$ = share of fires in position bin $b$ (64 bins × 8 positions); $TV = \tfrac12\sum_b |w_b - 1/64|$ (0 = position-independent, 1 = fully displaced from uniform). Raw TV is inflated for components with few fires, so we subtract the Monte-Carlo expectation for the same number of uniform fires: **excess TV** (≈ 0 for flat components). Components with excess TV ≥ 0.5 are **heavily** position-dependent (bold); the tables list everything ≥ 0.25.

**Activation firings (control):** the same profile/metric for *activation* firings, $|a_c| > 1$ with $a_c = \|U_c\|\,(V_c\cdot\varphi)$ (the norm of the component's rank-one write into the q/k output) — columns `a-fires`, `a-excess TV` and their own heatmaps. For **L0** the module input $\varphi$ is a function of the token id alone (post-embedding RMSNorm; position enters attention only via RoPE, downstream of q/k-space), so L0 activation profiles are position-independent by architecture — a null control for the pipeline. For L1 the input carries whatever layer 0 wrote, so positional activations are possible.

**Position 0 is special** (visible in the median-pos column): q components never have CI there — with only one visible key, softmax over a single logit is constant, so the query has zero causal effect at position 0 (holds for all layers: ≤ 37 stray q fires at pos 0 model-wide vs thousands at pos 1). k components show the mirror image: key 0 is read by every later query (the attention-sink site), so k CI is *largest* at position 0.

**Heatmaps:** four per matrix. *Firing* heatmaps show fire-count density relative to uniform (a value of 10 at a position = 10× more of the component's fires land there than a flat profile would put there). *Mean* heatmaps show the per-position mean of CI ($S_p/4000$) resp. $|a_c|$ ($A_p/4000$), divided by the component's overall mean — the same ratio-to-flat reading, but weighted by magnitude instead of binarized. Rows are sorted by CI excess TV in all four. Each figure's color scale spans its own data range (the scales are **not** comparable across figures).

## h.3.attn.q_proj

36 alive components; 3 heavy, 0 moderate. The remaining 33 have excess TV ≤ 0.06 (median 0.01) — position-independent within noise. Activation firings: max a-excess TV = 0.02 — no component's *activation* is position-dependent.

| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 | a-fires | a-excess TV | label |
|---|---|---|---|---|---|---|---|---|---|
| **261** | 2369 | 0.93 | 0.86 | 1 | 94 | 95 | 506581 | 0.00 | polysemantic component |
| **482** | 13214 | 0.86 | 0.83 | 2 | 88 | 89 | 983905 | 0.02 | fires negatively on endoftext, positively on general text |
| **458** | 4812 | 0.66 | 0.61 | 1 | 68 | 69 | 621989 | 0.01 | fires on document boundaries and <|endoftext|> |

![h.3.attn.q_proj CI firing profiles](../hide/figures/L3-pos-ci/q.png)

![h.3.attn.q_proj mean CI profiles](../hide/figures/L3-pos-ci/q_mean.png)

![h.3.attn.q_proj activation firing profiles](../hide/figures/L3-pos-ci/q_act.png)

![h.3.attn.q_proj mean activation profiles](../hide/figures/L3-pos-ci/q_act_mean.png)

## h.3.attn.k_proj

60 alive components; 14 heavy, 2 moderate. The remaining 44 have excess TV ≤ 0.15 (median 0.01) — position-independent within noise. Activation firings: max a-excess TV = 0.03 — no component's *activation* is position-dependent.

| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 | a-fires | a-excess TV | label |
|---|---|---|---|---|---|---|---|---|---|
| **133** | 4059 | 0.98 | 0.94 | 0 | 100 | 100 | 512027 | 0.00 | first token in the context window |
| **394** | 4060 | 0.98 | 0.93 | 0 | 100 | 100 | 637589 | 0.01 | fires on the first token of a sequence |
| **79** | 4058 | 0.98 | 0.93 | 0 | 100 | 100 | 398397 | 0.01 | first token of the sequence |
| **208** | 4061 | 0.98 | 0.93 | 0 | 100 | 100 | 525584 | 0.00 | first token in a sequence |
| **132** | 4061 | 0.98 | 0.93 | 0 | 100 | 100 | 535046 | 0.00 | first token of a sequence |
| **5** | 4062 | 0.98 | 0.93 | 0 | 100 | 100 | 627832 | 0.01 | fires on the first token of a sequence |
| **492** | 4064 | 0.98 | 0.93 | 0 | 100 | 100 | 786399 | 0.01 | first token in sequence |
| **289** | 5201 | 0.98 | 0.93 | 0 | 99 | 99 | 1190842 | 0.01 | fires on the first token of a sequence |
| **75** | 9394 | 0.80 | 0.77 | 1 | 80 | 83 | 599585 | 0.01 | first token in sequence or document boundary |
| **180** | 6719 | 0.76 | 0.72 | 0 | 77 | 78 | 1111348 | 0.02 | first sequence token and endoftext tokens |
| **490** | 6108 | 0.76 | 0.72 | 0 | 77 | 78 | 651589 | 0.01 | sequence boundaries and first tokens |
| **336** | 5949 | 0.75 | 0.71 | 0 | 77 | 78 | 1005953 | 0.02 | sequence beginnings and document boundaries |
| **65** | 5472 | 0.73 | 0.69 | 0 | 75 | 76 | 703932 | 0.01 | fires on sequence starts and document boundaries |
| **277** | 63 | 0.98 | 0.61 | 0 | 100 | 100 | 392509 | 0.00 | first token of a sequence |
| 487 | 25441 | 0.37 | 0.35 | 68 | 20 | 39 | 1196322 | 0.03 | noun and subject tokens prior to relational prepositions |
| 475 | 17220 | 0.27 | 0.25 | 125 | 28 | 33 | 1014438 | 0.02 | fires on code identifiers and sequence starts |

![h.3.attn.k_proj CI firing profiles](../hide/figures/L3-pos-ci/k.png)

![h.3.attn.k_proj mean CI profiles](../hide/figures/L3-pos-ci/k_mean.png)

![h.3.attn.k_proj activation firing profiles](../hide/figures/L3-pos-ci/k_act.png)

![h.3.attn.k_proj mean activation profiles](../hide/figures/L3-pos-ci/k_act_mean.png)
