# Models and decompositions

Naming reference for all target models and VPD decompositions in this project.
(C/D/E named 2026-09-08; old/new A/new B named 2026-09-02.)

## Target models

| Model | Run id | Architecture | Trained | Local location |
|---|---|---|---|---|
| pile_4l | `t-9d2b8f02` | 4L, d=768, 6 heads, tied embeddings, plain causal attention | Pile, 100k steps | `prev_paper/models/pile_4layer/` |
| simple_2l | `gf6rbga0` | 2L, d=192, 6 heads, tied embeddings, plain causal attention | SimpleStories, 100k steps | `prev_paper/models/simplestories_2layer/` |
| sink seed 45 | `t-87f91319` | as pile_4l but **untied lm_head + learned attention sinks** (one logit/head), seed 45 | Pile, 100k steps | `sink-models/pretrain_cache/spd-t-87f91319/` |
| sink seed 46 | `t-75f6c439` | same, seed 46 | Pile, 100k steps | `sink-models/pretrain_cache/spd-t-75f6c439/` |
| freqtest47 | `t-freqtest47` | same recipe, seed 47, trained by us with the PUBLIC pretrainer (inv_freq trainable, trajectory recorded — tests the RoPE hypothesis; `own-pretrain/`) | Pile, 100k steps (launched 2026-09-18, Modal) | volume `vpd-4layer`: `/sink-models/pretrain_cache/spd-t-freqtest47/` |

⚠ The sink models' recorded `rotary_base: 10000` does **not** match their training-time RoPE — loading as configured (incl. the public PR #1002 JAX loader) gives NLL ≈ 8.1 instead of the logged val 2.65. Use `load.load_sink(seed)` (corrected spectrum installed by default); full story in `sink-models/rope_report.md`. **But (2026-09-18, `sink-models/rope_decomposition_check.md`): the C/D/E decompositions — training forward, logged evals, and shipped harvests — themselves ran the textbook base-10000 forward**, so for anything computed *through the decompositions* (CI, co-CI, …) the public loader as-is is the correct pairing; C/D/E are decompositions of the mis-loaded ~8-NLL model, not of the real sink LMs.

## Decompositions

| Name | Run id | Decomposes | Steps | Components | Alive | Notes | Local location |
|---|---|---|---|---|---|---|---|
| old | `s-55ea3f9b` | pile_4l (`t-9d2b8f02`) | 400k | 38,912 | 9,973 | the paper's main decomposition; Torch format; harvest.db + interp.db available | `prev_paper/models/pile_4layer/vpd_decomposition_s-55ea3f9b/` |
| simple_2l | `s-eab2ace8` | simple_2l (`gf6rbga0`) | 400k | 7,104 | 6,535 | the paper's SimpleStories decomposition; Torch format; harvest.db + interp.db available | `prev_paper/models/simplestories_2layer/vpd_decomposition_s-eab2ace8/` |
| new A | `p-8383f5e5` | pile_4l (`t-9d2b8f02`) | 800k | 36,864 | 16,919 | minimality coeff 6.6e-5; JAX/Orbax format | `new-decomps/runs/p-8383f5e5/` |
| new B | `p-4d9a6a12` | pile_4l (`t-9d2b8f02`) | 800k | 36,864 | 23,002 | minimality coeff 6.6e-6 (10× weaker sparsity pressure than new A) | `new-decomps/runs/p-4d9a6a12/` |
| C | `p-d60af588` | sink seed 45 (`t-87f91319`) | 100k | 36,864 | 19,403 | new-A recipe (coeff 6.6e-5) at batch 256; decomposition seed 0 | `sink-models/runs/p-d60af588/` |
| D | `p-fecd6a6b` | sink seed 45 (`t-87f91319`) | 100k | 36,864 | 19,433 | identical to C except decomposition seed 1 (seed-consistency pair with C) | `sink-models/runs/p-fecd6a6b/` |
| E | `p-bd411e35` | sink seed 46 (`t-75f6c439`) | 100k | 36,864 | 21,668 | same recipe, decomposition seed 0 | `sink-models/runs/p-bd411e35/` |
| **F** | `p-c45e0001` | sink seed 45 (`t-87f91319`) **with the CORRECTED (fitted) RoPE forward** | 100k | 36,864 | — (in training) | our re-run of C's exact recipe with the true spectrum installed — the "decomposition of the real sink LM" C should have been; trained by us on Modal 2026-09-18 (`redo-decomps/`; substrate deviations from C's config documented in `redo-decomps/hide/config_ropefix_C.yaml`'s header; ignore volume run dirs `p-c45e0000/2/3/4/5/6/7/8/9/a/b` — dead smoke tests) | volume `vpd-4layer`: `/sink-models/runs/p-c45e0001/` |

Alive = mean CI > 1e-6. For old and simple_2l this is the harvest-run mean (20,000 batches × 32 seqs; the paper's definition). For new A/new B no harvest exists — the mean is over our 4,000-cached-Pile-row sample (2.05M tokens), a proxy for the same definition; C/D/E's counts are the same sample proxy (their 10M-token harvests arrived 2026-09-17 and agree, Spearman 0.96–0.9998 per matrix).
