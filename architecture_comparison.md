# Architecture comparison: Pile 4L vs SimpleStories 2L

Both models are the same `LlamaSimpleMLP` design (pre-RMSNorm blocks, rotary embeddings with
$d_{\text{rot}} = d_{\text{head}}$, GELU MLP, no biases, tied embedding/unembedding,
$d_{\text{mlp}} = 4\,d_{\text{model}}$). Configs: `models/*/target_model_*/model_config.yaml`.

| | Pile 4L (`t-9d2b8f02`) | SimpleStories 2L (`gf6rbga0`) |
|---|---|---|
| Layers | 4 | 2 |
| $d_{\text{model}}$ | 768 | 192 |
| Heads | 6 | 6 |
| KV heads | 6 (= MHA) | 3 (true GQA, 2 heads share each KV) |
| Head dim | 128 | 32 |
| MLP width | 3072 | 768 |
| Context | 512 | 512 |
| Vocab | 50,277 (GPT-NeoX) | 4,019 (SimpleStories GPT-2) |
| VPD subcomponents (total $C$) | 38,912 | 7,104 |
| Alive subcomponents | ~9,800 (paper: 9,972) | ~6,500 |
| Alive fraction | 25% | 91% |

## VPD subcomponent counts per matrix

Dictionary sizes $C$ are the same for every layer (from each decomposition's `final_config.yaml`,
`module_info`); overcompleteness = $C / \operatorname{rank}(W) = C / \min(d_{\text{in}}, d_{\text{out}})$.

**Alive** uses the paper's definition: mean causal importance $> 10^{-6}$ over the training
distribution. Since CI values are $O(1)$ when a component fires, this is equivalent to
"non-negligible CI ($\gtrsim 0.01$) on at least one token per $\sim 10^6$ tokens" — verified
empirically: counts with $\max_x \mathrm{CI}_c(x) > 0.01$ agree with the mean-based counts almost
exactly. Alive counts computed here (`ComponentModel` CI function, `lower_leaky`, over 100 training
sequences ≈ 51k Pile / 27k SimpleStories tokens); they reproduce the paper's per-layer Pile table
(3709/848/1943/3472, total 9,972) to within ~1.5% (undercounting slightly, since components rarer
than the sample size are missed), and the paper's layer-1 attention counts Q/K/V/O = 15/48/226/97
vs 14/49/223/97 here.

**Pile 4L** (decomposition `s-55ea3f9b`):

| Matrix | Shape $(d_{\text{in}} \times d_{\text{out}})$ | $C$ | $C/\text{rank}$ | Alive L0 | L1 | L2 | L3 | Alive total |
|---|---|---|---|---|---|---|---|---|
| `attn.q_proj` | $768 \times 768$ | 512 | 0.67 | 84 | 14 | 92 | 36 | 226 |
| `attn.k_proj` | $768 \times 768$ | 512 | 0.67 | 121 | 49 | 167 | 60 | 397 |
| `attn.v_proj` | $768 \times 768$ | 1024 | 1.33 | 468 | 223 | 503 | 249 | 1,443 |
| `attn.o_proj` | $768 \times 768$ | 1024 | 1.33 | 428 | 97 | 525 | 246 | 1,296 |
| `mlp.c_fc` | $768 \times 3072$ | 3072 | 4.0 | 1,371 | 248 | 289 | 1,031 | 2,939 |
| `mlp.down_proj` | $3072 \times 768$ | 3584 | 4.67 | 1,124 | 210 | 353 | 1,849 | 3,536 |
| **Sum** | | **9,728** ($\times 4$ layers $= 38{,}912$) | | **3,596** | **841** | **1,929** | **3,471** | **9,837** |

**SimpleStories 2L** (decomposition `s-eab2ace8`):

| Matrix | Shape $(d_{\text{in}} \times d_{\text{out}})$ | $C$ | $C/\text{rank}$ | Alive L0 | L1 | Alive total |
|---|---|---|---|---|---|---|
| `attn.q_proj` | $192 \times 192$ | 288 | 1.5 | 247 | 268 | 515 |
| `attn.k_proj` | $192 \times 96$ | 288 | 3.0 | 248 | 251 | 499 |
| `attn.v_proj` | $192 \times 96$ | 384 | 4.0 | 345 | 355 | 700 |
| `attn.o_proj` | $192 \times 192$ | 480 | 2.5 | 426 | 442 | 868 |
| `mlp.c_fc` | $192 \times 768$ | 1152 | 6.0 | 1,067 | 1,065 | 2,132 |
| `mlp.down_proj` | $768 \times 192$ | 960 | 5.0 | 882 | 903 | 1,785 |
| **Sum** | | **3,552** ($\times 2$ layers $= 7{,}104$) | | **3,215** | **3,284** | **6,499** |

Striking difference: the Pile decomposition leaves 75% of its capacity dead (especially in
layers 1–2 and Q/K everywhere), while the SimpleStories decomposition uses 91% of its
(relatively larger) dictionary — nearly every subcomponent is alive.

Differences beyond layer count:

1. **Uniform 4× scale-down**: $d_{\text{model}}$, head dim, and MLP width are all exactly $1/4$
   of the Pile model's.
2. **Attention sharing**: the Pile model has 6 KV heads for 6 query heads, so its grouped-query
   attention is effectively standard multi-head attention. The SimpleStories model has only
   3 KV heads, so pairs of query heads genuinely share keys and values — its $W_K, W_V$ are
   rectangular ($96 \times 192$) rather than square. Relevant when comparing attention
   subcomponents across the two models.
3. Different tokenizers/vocabularies.
