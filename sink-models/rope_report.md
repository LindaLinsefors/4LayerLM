# The sink models' RoPE does not match their recorded config

**TL;DR (2026-09-15).** The attention-sink base models (`t-87f91319` "sink seed 45", `t-75f6c439` "sink seed 46") were trained with a RoPE frequency spectrum that is *not* the `rotary_base: 10000` recorded in their `model_config.yaml`. Loaded with the recorded config — which is what the official sink-loader (PR #1002, commit `82a67f71c`) does — the models produce mean NLL **≈ 8.1** on Pile rows (near-garbage; confident-but-wrong at long range), while their WandB training logs show **val_loss 2.65**. We recovered the actual per-plane frequencies numerically from the weights; with them installed the models reach **NLL 2.79** on held-out cached Pile rows, *better* than the old `t-9d2b8f02` (2.84 on the same rows), consistent with the logged val margin. `load.load_sink()` applies the corrected spectrum by default.

**Consequence for prior work:** every analysis that ran the C/D/E decompositions' forward pass through the public loader (coci_{C,D,E}, top_tokens_{C,D,E}, pos_fires_sink_{C,D,E} → `C-pos-0/`, `read-in-cos/C_report.md`, the C/D/E parts of `compare-decomps/`, `coci-heatmaps/C/`) used the broken-RoPE forward. Residual streams at small positions (≲ 32–64 tokens, where all rotation angles are still small) are approximately right, so token-identity and pos-0 findings likely survive qualitatively — but long-range behavior, and any CI statistics aggregated over full 512-token rows, are quantitatively contaminated. **The co-authors should be asked for the internal RoPE convention** (see "What we know about the true convention" below) — and their own harvests/analyses of C/D/E may have the same problem if they use the public loader path.

## Symptom and verification chain

1. A local Torch port of the sink architecture (untied `lm_head`, per-head learned sink logit $s_h$: $\mathrm{softmax}$ over $[\,qk\text{-logits},\,s_h\,]$ with the sink column dropped — the GPT-OSS equation) gave mean NLL **8.105** on the first 8 cached Pile rows. The old pile_4l model gives 2.945 on the same rows.
2. The official JAX loader (`open_jax_run` at PR #1002) gives **8.108** on the same rows (`hide/verify_torch_port_modal.py`) — the port and the official loader agree to 3 decimals; both are "correct" implementations of the *recorded* config.
3. Greedy generation from short prompts is perfectly coherent, and top-1 accuracy is fine at short range: per-position NLL is 3.8/3.7/3.4 for positions < 64 but rises to **9.3** at positions 256–511. Entropy rises only mildly (3.9 → 4.9): the model is *confident but wrong* at long range — the signature of a positional-encoding mismatch, not a broken checkpoint.
4. WandB ground truth (`goodfire/spd/t-87f91319`, `t-75f6c439`): both runs finished at **val_loss ≈ 2.65** — the checkpoints are good. Their configs record `rotary_base: 10000`, so the training *code* (internal commit `37519f99`, not in the public repo) must have interpreted RoPE differently than its recorded config. The public "port" commits (`564490e`…`82a67f71c`) state the rotary convention as port constants rather than recording it from the run.

## Ruling out the standard families (`hide/rope_sweep.py`, `hide/gptoss_rope_test.py`)

NLL by position band (8 rows), selected variants:

| variant | 0–32 | 32–128 | 128–256 | 256–511 |
|---|---|---|---|---|
| as configured (rotate-half, base 1e4) | 3.85 | 4.99 | 8.27 | 9.73 |
| adjacent-pair pairing, base 1e4 | 7.23 | 9.82 | 10.08 | 10.47 |
| mixed layout/pairing (both directions) | 5–6.6 | 9.3–10 | — | — |
| no RoPE | 5.31 | 8.55 | 9.69 | 10.45 |
| partial rotary (rd 32/64/96, θ up to 1.5e6) | ≥ 4.5 | ≥ 6.3 | — | — |
| sliding window 64 (+ base 1e4) | 3.85 | 3.18 | 3.49 | 3.49 |
| plain base 1e6 | 3.77 | 3.07 | 3.00 | 3.24 |
| plain θ 150000 | 3.73 | 3.01 | 3.03 | 8.32 |
| YaRN θ150k, factor 4, orig-ctx 4096, no mscale | 3.74 | 3.02 | 2.94 | 3.05 |
| **old pile_4l on the same rows (reference)** | 3.75 | 2.88 | 2.83 | 2.93 |

QK-norm and logit softcap: ruled out / no effect. The per-band optimal base *grows with distance* — the true spectrum is not geometric in any base.

## Recovering the spectrum (`hide/fit_rope_freqs.py`)

Treat the 64 per-plane frequencies $\omega_p$ (rotate-half pairing, shared across heads and layers, angle $=\mathrm{pos}\cdot\omega_p$) as free parameters and minimize next-token NLL by Adam (48 train / 16 held-out cached rows, init = the best YaRN hand fit):

- seed 45: val NLL 3.35 (base 1e6) → 2.97 (YaRN init) → **2.788** after fitting;
- seed 46: → **2.798**;
- old pile_4l on the same 16 val rows: **2.839**.

So *pure rotate-half RoPE with the right spectrum fully explains the checkpoints* — no other architectural difference is needed, and the corrected sink models beat the old model by about the margin their training logs promised (2.65 vs 2.71).

**Cross-seed validation** (`hide/cross_seed_check.py`): each seed's fitted spectrum transferred to the *other* seed's model costs ≤ 0.018 nats vs its own fit — the recovered spectrum is the shared training-time convention, not per-model noise. The seed-averaged spectrum (`hide/cache/fitted_freqs_avg.npz`) scores 2.7882 / 2.7982 and is what `load_sink()` installs.

## What we know about the true convention

Well-constrained planes (0–26; seeds agree to ±10%), fitted $\omega_p$ in rad/token: 1.48, 1.06, 0.79, 0.66, 0.51, 0.45, 0.37, 0.30, 0.24, 0.22, 0.19, 0.156, 0.130, 0.105, 0.080, 0.065, 0.056, 0.045, 0.036, 0.026, 0.017, 0.014, 0.0055, 0.0053, 0.0043, 0.0029, 0.0041. Planes ≳ 27 are too slow to be constrained by a 512 context (fits diverge; any sufficiently small value works).

Odd fingerprints no standard scheme reproduces:
- $\omega_0 \approx 1.48 \ne 1$ (both seeds move it up from a 1.0 init) — every textbook convention has $\omega_0 = 1$;
- mean slope ≈ geometric base $1.6\times10^6$, but pure geometric ($A\,\theta^{-p/64}$, with or without the 1.48 scale) costs +0.3 nats vs the fit;
- an abrupt ≈3× frequency drop between planes 21 and 22 (λ crosses ≈ 500 tokens ≈ n_ctx) — YaRN-by-parts-flavored, but no (θ, factor, orig-ctx, β) combination tested matches (best hand fit: θ150k/f4/oc4096, +0.18 nats vs fitted).

θ = 150000 matching the fast planes and the sink mechanism itself suggest the internal experiment borrowed GPT-OSS attention pieces; the exact spectrum is likely one more config knob away. **Asking the co-authors for `param_decomp/experiments/lm/pretrain/models/llama_simple_mlp.py` at internal commit `37519f99` (or just "what RoPE did task-1423 use?") would settle it exactly.** Until then the fitted spectrum is behaviorally equivalent at the ~0.01-nat level on real data.

## Files

- `hide/verify_torch_port_modal.py` — Modal JAX reference logits/NLL (→ `hide/cache/jax_reference.npz`).
- `hide/rope_sweep.py`, `hide/gptoss_rope_test.py` — convention family sweeps.
- `hide/fit_rope_freqs.py <seed>` — the per-plane fit (→ `hide/cache/fitted_freqs_{45,46}.npz`).
- `hide/cross_seed_check.py` — cross-seed transfer + closed-form candidates (→ `fitted_freqs_avg.npz`).
- `load.load_sink(seed, corrected_rope=True)` — loads the Torch port with the fitted spectrum installed.
