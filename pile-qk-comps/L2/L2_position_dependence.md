# L2 — q/k components with position-in-chunk-dependent causal importance

Per alive component of `h.2.attn.q_proj` / `h.2.attn.k_proj`: where in the 512-token chunk do its CI firings (CI > 0.1) happen? Over the 4,000 cached Pile rows every chunk position is seen exactly 4,000 times, and chunk boundaries fall at random points inside documents, so the token distribution is the same at every position (position 0 is *not* usually a document start — 74.6% of chunks contain no EOS). A component whose CI depends only on token/context therefore has a flat firing-position profile; any structure is genuine position dependence. Computed on Modal (`hide/pos_ci_compute_modal.py` → `hide/cache/pos_ci.npz`); tables/figures from `hide/pos_ci_report.py`.

**Metric:** profile $w_b$ = share of fires in position bin $b$ (64 bins × 8 positions); $TV = \tfrac12\sum_b |w_b - 1/64|$ (0 = position-independent, 1 = fully displaced from uniform). Raw TV is inflated for components with few fires, so we subtract the Monte-Carlo expectation for the same number of uniform fires: **excess TV** (≈ 0 for flat components). Components with excess TV ≥ 0.5 are **heavily** position-dependent (bold); the tables list everything ≥ 0.25.

**Activation firings (control):** the same profile/metric for *activation* firings, $|a_c| > 1$ with $a_c = \|U_c\|\,(V_c\cdot\varphi)$ (the norm of the component's rank-one write into the q/k output) — columns `a-fires`, `a-excess TV` and their own heatmaps. For **L0** the module input $\varphi$ is a function of the token id alone (post-embedding RMSNorm; position enters attention only via RoPE, downstream of q/k-space), so L0 activation profiles are position-independent by architecture — a null control for the pipeline. For L1 the input carries whatever layer 0 wrote, so positional activations are possible.

**Position 0 is special** (visible in the median-pos column): q components never have CI there — with only one visible key, softmax over a single logit is constant, so the query has zero causal effect at position 0 (holds for all layers: ≤ 37 stray q fires at pos 0 model-wide vs thousands at pos 1). k components show the mirror image: key 0 is read by every later query (the attention-sink site), so k CI is *largest* at position 0.

**Heatmaps:** four per matrix. *Firing* heatmaps show fire-count density relative to uniform (a value of 10 at a position = 10× more of the component's fires land there than a flat profile would put there). *Mean* heatmaps show the per-position mean of CI ($S_p/4000$) resp. $|a_c|$ ($A_p/4000$), divided by the component's overall mean — the same ratio-to-flat reading, but weighted by magnitude instead of binarized. Rows are sorted by CI excess TV in all four. Each figure's color scale spans its own data range (the scales are **not** comparable across figures).

## h.2.attn.q_proj

92 alive components; 0 heavy, 0 moderate. The remaining 92 have excess TV ≤ 0.16 (median 0.06) — position-independent within noise. Activation firings: max a-excess TV = 0.04 — no component's *activation* is position-dependent.

| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 | a-fires | a-excess TV | label |
|---|---|---|---|---|---|---|---|---|---|

![h.2.attn.q_proj CI firing profiles](../hide/figures/L2-pos-ci/q.png)

![h.2.attn.q_proj mean CI profiles](../hide/figures/L2-pos-ci/q_mean.png)

![h.2.attn.q_proj activation firing profiles](../hide/figures/L2-pos-ci/q_act.png)

![h.2.attn.q_proj mean activation profiles](../hide/figures/L2-pos-ci/q_act_mean.png)

## h.2.attn.k_proj

167 alive components; 30 heavy, 3 moderate. The remaining 134 have excess TV ≤ 0.19 (median 0.03) — position-independent within noise. Activation firings: max a-excess TV = 0.04 — no component's *activation* is position-dependent.

| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 | a-fires | a-excess TV | label |
|---|---|---|---|---|---|---|---|---|---|
| **449** | 4059 | 0.98 | 0.94 | 0 | 100 | 100 | 787606 | 0.01 | first token of the sequence |
| **67** | 4060 | 0.98 | 0.94 | 0 | 100 | 100 | 894610 | 0.02 | fires mostly on sequence boundaries or early tokens |
| **134** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 652893 | 0.03 | first token of the sequence |
| **261** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 643649 | 0.00 | first token of sequence |
| **487** | 4060 | 0.98 | 0.93 | 0 | 100 | 100 | 806079 | 0.00 | first token in sequence |
| **511** | 4060 | 0.98 | 0.93 | 0 | 100 | 100 | 679216 | 0.01 | first token of a sequence |
| **88** | 4061 | 0.98 | 0.93 | 0 | 100 | 100 | 646164 | 0.02 | fires at the beginning of sequences |
| **243** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 759199 | 0.02 | first token in a sequence |
| **464** | 4060 | 0.98 | 0.93 | 0 | 100 | 100 | 937839 | 0.01 | fires on the first token of a sequence |
| **86** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 740563 | 0.00 | first token in a sequence |
| **503** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 670594 | 0.01 | first token of sequence |
| **509** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 608229 | 0.00 | first token in a sequence |
| **458** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 709757 | 0.03 | fires on the first token of a sequence |
| **68** | 4059 | 0.98 | 0.93 | 0 | 100 | 100 | 656456 | 0.01 | fires on the first token of the sequence |
| **413** | 4060 | 0.98 | 0.93 | 0 | 100 | 100 | 670797 | 0.01 | first token in sequence |
| **196** | 4083 | 0.98 | 0.93 | 0 | 99 | 99 | 563102 | 0.00 | fires on the first token of the sequence |
| **271** | 4083 | 0.98 | 0.93 | 0 | 99 | 99 | 861800 | 0.03 | fires on the first token of a sequence |
| **259** | 4085 | 0.98 | 0.93 | 0 | 99 | 99 | 930411 | 0.04 | fires near tables, formatting, or random tokens |
| **453** | 4089 | 0.98 | 0.93 | 0 | 99 | 99 | 682694 | 0.01 | fires on the first token of sequences |
| **490** | 4302 | 0.93 | 0.88 | 0 | 94 | 94 | 818492 | 0.01 | fires near the beginning of sequences or documents |
| **3** | 250 | 0.96 | 0.76 | 0 | 97 | 97 | 366750 | 0.01 | first token of the sequence |
| **23** | 5948 | 0.75 | 0.71 | 0 | 77 | 78 | 840046 | 0.02 | sequence start and document boundary tokens |
| **131** | 5971 | 0.75 | 0.71 | 0 | 77 | 78 | 953127 | 0.02 | attention sink (first token and endoftext) |
| **99** | 5975 | 0.75 | 0.71 | 0 | 77 | 78 | 1130242 | 0.01 | fires on sequence start and endoftext tokens |
| **508** | 6039 | 0.75 | 0.71 | 0 | 77 | 77 | 1025355 | 0.03 | fires on first token of sequence or document |
| **446** | 6030 | 0.75 | 0.71 | 0 | 77 | 77 | 986962 | 0.02 | sequence start and endoftext tokens (attention sink) |
| **73** | 5932 | 0.75 | 0.71 | 0 | 77 | 77 | 903841 | 0.02 | sequence start or document boundary token |
| **204** | 5994 | 0.75 | 0.71 | 0 | 77 | 77 | 788024 | 0.01 | fires near the start of a sequence or after document boundaries |
| **444** | 5417 | 0.74 | 0.70 | 0 | 75 | 76 | 772957 | 0.02 | first token of sequence and document boundaries |
| **321** | 5472 | 0.73 | 0.69 | 0 | 75 | 76 | 682133 | 0.02 | sequence start and <|endoftext|> tokens |
| 39 | 10075 | 0.40 | 0.37 | 64 | 42 | 45 | 769030 | 0.00 | beginning of sequence and indentation tokens |
| 12 | 37598 | 0.31 | 0.30 | 96 | 25 | 36 | 1202194 | 0.02 | fires at the start of sequences or documents |
| 306 | 15661 | 0.29 | 0.27 | 139 | 31 | 34 | 1088270 | 0.01 | sequence boundaries and math variables/numbers |

![h.2.attn.k_proj CI firing profiles](../hide/figures/L2-pos-ci/k.png)

![h.2.attn.k_proj mean CI profiles](../hide/figures/L2-pos-ci/k_mean.png)

![h.2.attn.k_proj activation firing profiles](../hide/figures/L2-pos-ci/k_act.png)

![h.2.attn.k_proj mean activation profiles](../hide/figures/L2-pos-ci/k_act_mean.png)
