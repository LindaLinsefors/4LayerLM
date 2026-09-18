# What RoPE were the sink decompositions (C/D/E) trained with?

**TL;DR (2026-09-18).** Follow-up to [rope_report.md](rope_report.md), which is **re-verified** below (fresh held-out data, same conclusions). The new finding answers "what RoPE was used when *training the components*":

**The C/D/E decompositions — their training forward, their WandB eval metrics, and the shipped 10M-token harvests — all ran the target with textbook base-10000 RoPE, i.e. exactly the "broken" forward the public loader builds.** The RoPE mismatch is *between the internal pretrain code and the internal decomposition stack*: the pretrain code trained the base models with the non-standard spectrum (recovered numerically in rope_report.md), while the decomposition/harvest code interpreted the recorded `rotary_base: 10000` literally. Consequently:

- **C/D/E are decompositions of the mis-loaded model** — the sink-pretrained *weights* run through a textbook-RoPE forward, an LM with NLL ≈ 7.9–8.1 on Pile (confident-but-wrong at long range) — *not* of the val-loss-2.65 language model the pretraining produced.
- **The public loader is the *right* loader for the decompositions.** All pre-2026-09-15 C/D/E CI caches (coci, top_tokens, pos_fires, C-pos-0, compare-decomps, coci-heatmaps/C, cos-sim C co-CI) computed through the public loader are *internally consistent with the decompositions' own training semantics* — the "contamination" warning inverts. Conversely, the 2026-09-17 standard "install the fitted spectrum for all C/D/E JAX forwards" is correct for studying the *base LMs* but **mismatched for studying the decompositions**: under the fitted forward the decomposition loses its sparsity (per-site L0 roughly doubles, "alive" components 19,409 → 29,211 for C) because minimality was optimized against the configured forward.
- **Scientific caveat that does NOT invert:** any conclusion of the form "the sink-model decompositions lack the emergent boundary/sink machinery" now has a confound — C/D/E describe a model whose positional structure is broken, so differences vs the old decomposition may reflect the mis-load rather than the built-in-sink architecture. Studies of the *actual* sink LMs (forwards via `load.load_sink`, attention patterns, copying, probes) are unaffected and should keep using the fitted spectrum.

The co-authors should be told: if their intent was to decompose the trained sink LMs, runs `p-d60af588` / `p-fecd6a6b` / `p-bd411e35` and their harvests decomposed/analyzed a mis-loaded model. (And the question "what RoPE did task-1423's *pretrain* use?" still needs their answer — see §4.)

## 1. Re-verification of rope_report.md

All checks reproduce, on data the original fit never touched (cached Pile rows 64:128; the fit used 0:48 train / 48:64 val). `hide/verify_rope_fresh.py`:

| forward | sink45 | sink46 | pile_4l (ref) |
|---|---|---|---|
| as configured (rotate-half, base $10^4$) | 7.806 | 7.620 | — |
| fitted spectrum, seed-averaged | 2.504 | 2.499 | — |
| fitted spectrum, own-seed fit | 2.502 | 2.497 | — |
| pile_4l on the same rows | — | — | 2.553 |

The corrected sink models beat pile_4l by ≈ 0.05 nats — the logged val margin (2.65 vs 2.71). The seed-averaged spectrum costs ≤ 0.002 nats vs each seed's own fit (cross-seed transfer re-confirmed). The local Torch port matches the official JAX loader on the recorded config: mean NLL 8.105 vs 8.108, per-row |Δ| ≤ 0.008. No sign of overfitting to the fit's validation rows.

## 2. Evidence: the decomposition stack ran textbook base-10000

Three independent lines, in increasing order of strength.

### 2a. Code/config circumstantials

- The decomposition launch configs record **no RoPE fields at all**; the target is `kind: pretrained`, `model_class: param_decomp.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP`, resolved by the *decomposition stack's own* target adapter (the public port `param_decomp/targets/llama_simple_mlp.py` mirrors it: `plain_rope_inv_freq`, textbook). The dotted class string is an identifier, never imported (per the repo docs), so the pretrain code's actual RoPE never enters the decomposition.
- The CI function's rotary buffer is stored in the checkpoint (`ci_fns.inv_freq`, grad-norm 0 on WandB). Extracted from all five JAX decompositions (C/D/E/newA/newB, `hide/rope_decomp_check_modal.py` → `hide/cache/ci_fn_invfreq.npz`): **byte-identical across all five and equal to bf16-quantized textbook** $10000^{-p/64}$ (max deviation 0.0012 = bf16 rounding). The internal stack's rotary generator is textbook; the weird spectrum lives only in the pretrain attention.
- Pretrain and decomposition runs used the same environment (identical `requirements.txt`: jax 0.11.1, `param-decomp==0.0.1` + internal `param-decomp-goodfire==0.0.1`) within ~30 h of each other — same internal code generation, two different attention implementations.
- The Sep-2026 pretrain configs record only `rotary_base: 10000` (+ `attention_sinks`), having *dropped* the `rotary_dim`/`rotary_adjacent_pairs` fields the Feb-2026 `t-9d2b8f02` config carried — the pretrain model schema was rewritten, consistent with RoPE details becoming (non-standard) code constants there.

### 2b. The shipped harvests match the configured forward

Per-component sample mean CI over the 4,000 cached Pile rows, recomputed under both spectra with identical code (`hide/rope_decomp_check_modal.py` → `hide/cache/mean_ci_{fitted,configured}_{C,D,E}.npz`), and compared per site against `harvest.sqlite`'s `mean_causal_importance` (the co-authors' 10M-token pipeline). Spearman ρ, summarized (`hide/rope_harvest_compare.py` prints all 72 rows):

| run | mean ρ configured | mean ρ fitted | min ρ configured | min ρ fitted |
|---|---|---|---|---|
| C | **0.962** | 0.801 | 0.820 | 0.240 (h.1.attn.q_proj) |
| D | **0.967** | 0.810 | 0.847 | 0.275 |
| E | **0.975** | 0.862 | 0.868 | 0.506 |

Configured wins on **all 72 sites**. The residual gap from ρ = 1 under configured is data-sample difference (10M held-out tokens vs our 2.05M train-distribution rows) — the configured columns reproduce the pre-existing `coci_{C,D,E}.npz` caches to ~10⁻³, so the earlier harvest-widget agreement was already this signal. The harvests ran the broken forward.

### 2c. The decomposition runs' own logged eval metrics reproduce only under the configured forward

The WandB runs logged `CEandKLLosses` at step 100k (their held-out `pile_neox_tok_512_val`). Reproduced with the public eval semantics (`hide/rope_ce_kl_modal.py`, mirrors `experiments/lm/eval.py::make_ce_kl_scorer`; our rows 0:128, so ~0.01–0.07 data-set differences expected) — run C:

| metric | logged (C) | reproduced, configured | reproduced, fitted |
|---|---|---|---|
| KL(target ‖ CI-masked) | 0.618 | **0.615** | 1.484 |
| KL(target ‖ rounded-masked) | 0.592 | **0.589** | 1.336 |
| KL(target ‖ unmasked) | 0.0061 | **0.0062** | 0.0014 |
| KL(target ‖ zero-masked) | 10.09 | **10.18** | 11.70 |
| CE(CI-masked) − CE(target) | −0.314 | **−0.342** | +1.407 |
| CE(rounded) − CE(target) | −0.385 | **−0.420** | +1.251 |
| CE(unmasked) − CE(target) | −0.025 | **−0.026** | +0.001 |

Run E matches its own logged values the same way (logged 0.589/0.562/10.13 vs configured-reproduced 0.599/0.572/10.14). Two details are independently diagnostic:

- **CE(target) on their eval ≈ 7.85 (C) / 7.66 (E)** under the configured forward — the trainer's own target was the ~8-NLL broken LM, matching the *negative* logged CE differences: CI-masking *improves* the target by 0.2–0.4 nats, which is essentially impossible for a healthy 2.65-CE target but natural for a confidently-wrong one (ablation reduces miscalibrated confidence).
- Under the fitted forward the decomposition stops being sparse or faithful-in-behavior: KL(CI-masked) triples, CE difference flips to +1.4, per-site L0(CI > 0.1) roughly doubles (e.g. h.0.mlp.c_fc 10.4 → 26.0), sample-alive components 19,409 → 29,211 (C). Minimality/mechanistic-faithfulness were optimized under the configured forward.

## 3. Practical consequences for this project

1. **Studying the decompositions C/D/E (CI, co-CI, harvests, activation examples, component circuits):** use the **public loader as-is** (configured RoPE). Everything computed that way — the pre-2026-09-15 caches plus `activation_examples.html`, the write-out C co-CI Gram, the harvest widget — is the *correct* pairing with the decompositions' semantics. Their earlier "broken-RoPE contamination" flags should be read instead as: *consistent with the decomposition, but describing the mis-loaded model*.
2. **Studying the base sink LMs themselves:** unchanged — `load.load_sink(seed)` with the fitted spectrum (rope_report.md stands in full).
3. The one C/D/E artifact computed under the 2026-09-17 fitted-spectrum standard is `components/hide/cache/typical_act_C.npz` (CI-weighted typical activations feeding `components/C/q_dot_k*/`). Its CI weighting mixes fitted-forward CI with a decomposition trained on the configured forward — recompute under the configured forward if those figures are load-bearing.
4. **Interpretation caveat (the part that cannot be fixed by choosing a loader):** C/D/E decompose an LM that was never trained as such. Findings like "C has no EOS/boundary read-in cluster" (cos-sim), "the token-independent pos-0 sink machinery is essentially gone" (pos_fires analyses) are statements about the broken forward's mechanisms and are confounded as evidence about built-in-sink *architecture*. Cross-model comparisons (old/newA/newB vs C/D/E) should carry this caveat until re-trained decompositions exist.
5. **For the co-authors:** their decomposition training and harvests of task-1423 mis-loaded the sink targets (textbook RoPE over weights trained with a non-standard spectrum). If they want decompositions of the real sink LMs, the runs need re-training with the pretrain-time RoPE — which only they can name exactly (internal commit `37519f99`).

## 4. Status of the true pretrain spectrum

Unchanged from rope_report.md — the fitted spectrum (`hide/cache/fitted_freqs_avg.npz`) remains the best available stand-in (behaviorally within ~0.01 nats). Two additions from this session:

- **No YaRN/NTK-family config can be the answer.** All standard schemes leave plane 0 unscaled: rotations of plane 0 in any original context ≫ β_fast, so the interpolation ramp is 1 there and $\omega_0 = 1$ exactly, for *every* (θ, factor, orig-ctx, β) combination. The fitted $\omega_0 \approx 1.48$ (both seeds independently, ±5%) is therefore structurally out of reach of that whole family — not a matter of untested parameter combinations. (Caveat: on integer positions any $\omega$ is observable only mod $2\pi$, and a temperature/mscale factor is outside the fit's model class; but no standard spectrum exceeds Nyquist, so aliasing shouldn't arise.)
- **Leading hypothesis: trainable (learned) `inv_freq` in the internal pretrain attention.** It would explain a smooth non-geometric spectrum, $\omega_0 > 1$, and near-identical spectra across two seeds (same data → similar optimum), and it fits the schema observation in §2a (RoPE fields dropped from the recorded config). The public safetensors export contains no `inv_freq` tensor (39 tensors, checked) — but an export that assumes buffers-are-derivable would drop exactly that. Testable by asking the co-authors, or against the original pretrain checkpoints (not on WandB: the pretrain runs' artifacts hold only history parquets).

## Files

- `hide/verify_rope_fresh.py` — §1 re-verification (local GPU, ~2 min).
- `hide/rope_decomp_check_modal.py` — Modal (11 containers): extracts `ci_fn.inv_freq` from all five decomposition checkpoints (→ `hide/cache/ci_fn_invfreq.npz`) and recomputes C/D/E per-component mean CI + fire counts under both spectra (→ `hide/cache/mean_ci_{fitted,configured}_{C,D,E}.npz`).
- `hide/rope_harvest_compare.py` — §2b table (local).
- `hide/rope_ce_kl_modal.py` — §2c reproduction (Modal, 4 containers; → `hide/cache/rope_ce_kl.npz`).
