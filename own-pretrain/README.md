# own-pretrain — sink LM "t-freqtest47" with trainable RoPE frequencies

Pretrains a NEW 4-layer Pile attention-sink language model, replicating the
co-authors' **sink seed 45** run (WandB `goodfire/spd/t-87f91319`) exactly except

- seed **47** instead of 45,
- the RoPE inverse-frequency **trajectory is recorded** throughout training.

## Purpose: test the trainable-inv_freq hypothesis

The seed-45/46 sink base models were trained with a RoPE spectrum that does not
match their recorded `rotary_base: 10000` (`sink-models/rope_report.md`); the true
spectrum was recovered numerically (`sink-models/hide/cache/fitted_freqs_avg.npz`,
e.g. $\omega_0 \approx 1.48$ vs textbook $\omega_0 = 1$). Leading hypothesis: the
internal pretrain code made `inv_freq` a *trainable* parameter.

**Finding from reading the public pretrainer** (`sink-models/param-decomp-sink/
param_decomp/pretrain/`, PR #1002 commit `82a67f71c`): the public JAX pretrainer
**already trains `inv_freq`** — it is a plain (non-static) array leaf of
`LlamaSimpleMLP` (`models.py`), `train.py` differentiates and updates every array
leaf via `eqx.filter(..., eqx.is_array)`, and the AdamW weight-decay mask is
`ndim >= 2`, so the 1-D `inv_freq` sits in the **undecayed** group (with the
RMSNorm gains and sink logits). No `stop_gradient` anywhere. So running the
public trainer as-is *is* the hypothesized internal behavior — this itself is
evidence for the hypothesis (the internal code is this code's torch/JAX sibling).

If the trained `inv_freq` (init textbook $10000^{-p/64}$, $p = 0..63$) converges
toward the fitted seed-45/46 spectrum ($\omega_0$ rising toward ~1.5, mid planes
slowing ~3×), the hypothesis is confirmed and this model is a fully-known sink LM
of Linda's own. Divergent final values would refute it.

## The run

- Recipe = `param_decomp/pretrain/configs/pile_llama_simple_mlp-4L-768-untied-sinks.yaml`
  (matches the t-87f91319 WandB config dump field-for-field): LlamaSimpleMLP 4L/768/6h,
  MLP 3072, n_ctx 512, vocab 50277, untied head, attention sinks, AdamW β=(0.9, 0.95),
  lr 3e-4 cosine → 3e-5, warmup 600, wd 0.1 (2-D weights only), grad clip 1.0,
  global batch 1024, 100k steps, bf16 compute / fp32 masters.
- Data: prestaged store `pile_neox_tok_512` (106 parquet shards of
  `danbraunai/pile-uncopyrighted-tok-shuffled`) on Modal volume `vpd-4layer`
  under `/data/sink-models/datasets/`; `val_data: null` = in-distribution
  monitoring on a reseeded schedule over the train shards (as in the internal run).
- Launcher: `hide/pretrain_modal.py` (Modal, 8×H100, resumable via orbax
  checkpoints + Modal retries). Smoke mode = 500 steps.

## Outputs (Modal volume `vpd-4layer`)

- `/sink-models/runs/t-freqtest47/inv_freq_trajectory.npz` — keys `steps` (N,),
  `inv_freq` (N, 64) float32, `init` (64,); one snapshot per 100 steps.
- `/sink-models/runs/t-freqtest47/metrics.jsonl` + wandb (Linda's entity, project
  `param-decomp`, group `own-pretrain`, run `t-freqtest47`): train/val loss and
  `inv_freq_0`, `inv_freq_drift_l2`, `inv_freq_drift_linf`, `inv_freq_logdrift_l2`.
- `/sink-models/runs/t-freqtest47/ckpts/` — orbax resume checkpoints (keep-last 2).
- `/sink-models/pretrain_cache/spd-t-freqtest47/` — `model_step_100000.safetensors`
  + `model_config.yaml`, loadable/decomposable exactly like `spd-t-87f91319`.

Monitor / fetch (PowerShell, always `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`):

```
modal app list                                   # is it still running
modal volume get vpd-4layer /sink-models/runs/t-freqtest47/inv_freq_trajectory.npz .
modal volume get vpd-4layer /sink-models/runs/t-freqtest47/metrics.jsonl .
```
