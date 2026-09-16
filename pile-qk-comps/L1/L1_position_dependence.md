# L1 — q/k components with position-in-chunk-dependent causal importance

Per alive component of `h.1.attn.q_proj` / `h.1.attn.k_proj`: where in the 512-token chunk do its CI firings (CI > 0.1) happen? Over the 4,000 cached Pile rows every chunk position is seen exactly 4,000 times, and chunk boundaries fall at random points inside documents, so the token distribution is the same at every position (position 0 is *not* usually a document start — 74.6% of chunks contain no EOS). A component whose CI depends only on token/context therefore has a flat firing-position profile; any structure is genuine position dependence. Computed on Modal (`hide/pos_ci_compute_modal.py` → `hide/cache/pos_ci.npz`); tables/figures from `hide/pos_ci_report.py`.

**Metric:** profile $w_b$ = share of fires in position bin $b$ (64 bins × 8 positions); $TV = \tfrac12\sum_b |w_b - 1/64|$ (0 = position-independent, 1 = fully displaced from uniform). Raw TV is inflated for components with few fires, so we subtract the Monte-Carlo expectation for the same number of uniform fires: **excess TV** (≈ 0 for flat components). Components with excess TV ≥ 0.5 are **heavily** position-dependent (bold); the tables list everything ≥ 0.25.

**Activation firings (control):** the same profile/metric for *activation* firings, $|a_c| > 1$ with $a_c = \|U_c\|\,(V_c\cdot\varphi)$ (the norm of the component's rank-one write into the q/k output) — columns `a-fires`, `a-excess TV` and their own heatmaps. For **L0** the module input $\varphi$ is a function of the token id alone (post-embedding RMSNorm; position enters attention only via RoPE, downstream of q/k-space), so L0 activation profiles are position-independent by architecture — a null control for the pipeline. For L1 the input carries whatever layer 0 wrote, so positional activations are possible.

**Position 0 is special** (visible in the median-pos column): q components never have CI there — with only one visible key, softmax over a single logit is constant, so the query has zero causal effect at position 0 (holds for all layers: ≤ 37 stray q fires at pos 0 model-wide vs thousands at pos 1). k components show the mirror image: key 0 is read by every later query (the attention-sink site), so k CI is *largest* at position 0.

**Heatmaps:** four per matrix. *Firing* heatmaps show fire-count density relative to uniform (a value of 10 at a position = 10× more of the component's fires land there than a flat profile would put there). *Mean* heatmaps show the per-position mean of CI ($S_p/4000$) resp. $|a_c|$ ($A_p/4000$), divided by the component's overall mean — the same ratio-to-flat reading, but weighted by magnitude instead of binarized. Rows are sorted by CI excess TV in all four. Each figure's color scale spans its own data range (the scales are **not** comparable across figures).

## h.1.attn.q_proj

15 alive components; 7 heavy, 0 moderate. The remaining 8 have excess TV ≤ 0.18 (median 0.05) — position-independent within noise. Activation firings: max a-excess TV = 0.02 — no component's *activation* is position-dependent.

| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 | a-fires | a-excess TV | label |
|---|---|---|---|---|---|---|---|---|---|
| **497** | 3924 | 0.98 | 0.93 | 1 | 100 | 100 | 374584 | 0.02 | repeating identical tokens or characters |
| **149** | 6499 | 0.97 | 0.93 | 2 | 99 | 99 | 620035 | 0.02 | generic component predicting common punctuation and stopwords |
| **268** | 1902 | 0.97 | 0.90 | 1 | 99 | 99 | 363620 | 0.01 | word fragments and compounds in specific phrases |
| **342** | 1483 | 0.98 | 0.90 | 1 | 100 | 100 | 308681 | 0.01 | predicts word ends and boundaries |
| **37** | 301 | 0.98 | 0.80 | 1 | 100 | 100 | 340477 | 0.01 | predicts prepositions, punctuation, and word continuations |
| **474** | 232 | 0.98 | 0.77 | 2 | 100 | 100 | 391378 | 0.01 | fires on period tokens |
| **271** | 53 | 0.98 | 0.56 | 1 | 100 | 100 | 383580 | 0.02 | commas and open parentheses |

![h.1.attn.q_proj CI firing profiles](../hide/figures/L1-pos-ci/q.png)

![h.1.attn.q_proj mean CI profiles](../hide/figures/L1-pos-ci/q_mean.png)

![h.1.attn.q_proj activation firing profiles](../hide/figures/L1-pos-ci/q_act.png)

![h.1.attn.q_proj mean activation profiles](../hide/figures/L1-pos-ci/q_act_mean.png)

## h.1.attn.k_proj

48 alive components; 10 heavy, 0 moderate. The remaining 38 have excess TV ≤ 0.04 (median -0.00) — position-independent within noise. Activation firings: max a-excess TV = 0.03 — no component's *activation* is position-dependent.

| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 | a-fires | a-excess TV | label |
|---|---|---|---|---|---|---|---|---|---|
| **315** | 6103 | 0.98 | 0.94 | 0 | 100 | 100 | 583209 | 0.01 | fires at the beginning of sequences |
| **339** | 5843 | 0.98 | 0.94 | 0 | 100 | 100 | 482676 | 0.02 | uninterpretable / polysemantic |
| **357** | 9066 | 0.97 | 0.93 | 1 | 98 | 100 | 637115 | 0.03 | miscellaneous text and punctuation tokens |
| **272** | 4007 | 0.98 | 0.93 | 0 | 100 | 100 | 331960 | 0.02 | fires on the first token of a sequence |
| **121** | 3375 | 0.98 | 0.93 | 0 | 100 | 100 | 487126 | 0.00 | fires on the first token of a sequence |
| **147** | 2841 | 0.98 | 0.92 | 0 | 100 | 100 | 445407 | 0.00 | early sequence tokens |
| **52** | 1143 | 0.98 | 0.89 | 1 | 100 | 100 | 730757 | 0.00 | alphanumeric tokens in acronyms, variables, and identifiers |
| **196** | 580 | 0.98 | 0.85 | 1 | 100 | 100 | 499239 | 0.00 | fires on numbers to predict punctuation and units |
| **318** | 97 | 0.98 | 0.66 | 1 | 100 | 100 | 342097 | 0.00 | fires on commas to predict conjunctions and transitions |
| **327** | 4191 | 0.66 | 0.61 | 1 | 67 | 69 | 389319 | 0.01 | sequence boundaries and document starts |

![h.1.attn.k_proj CI firing profiles](../hide/figures/L1-pos-ci/k.png)

![h.1.attn.k_proj mean CI profiles](../hide/figures/L1-pos-ci/k_mean.png)

![h.1.attn.k_proj activation firing profiles](../hide/figures/L1-pos-ci/k_act.png)

![h.1.attn.k_proj mean activation profiles](../hide/figures/L1-pos-ci/k_act_mean.png)
