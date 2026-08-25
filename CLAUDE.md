# CLAUDE.md — 4LayerLM

## Project purpose

Investigate the model studied in the previous paper, described in [prev_paper.md](prev_paper.md):
*"Interpreting Language Model Parameters"* (Goodfire/MATS, published May 5th 2026; the user, Linda Linsefors, is a co-author). The paper introduces **adVersarial Parameter Decomposition (VPD)**, a parameter-decomposition interpretability method, applied to a small language model: **67M parameters, 4 layers, trained on the Pile**.

Relevant external resource: the paper's code library — https://github.com/goodfire-ai/param-decomp

## Standing instructions

- **Keep this file up to date.** All Claude instances working on this project should continuously update this file with whatever seems useful about the state of the project (goals, decisions, findings, file layout, gotchas, environment details).
- **Use Modal.** The user has Modal access. Run compute on Modal whenever that is faster than running locally.
- **Code style:** write code that is both fast to run and easy for the user to read.
- **Math:** the user knows math well (physics background). When something can be explained with equations — in chat or in reports — use equations rather than only prose.

## Background: the model & method (from prev_paper.md)

- VPD decomposes each weight matrix into rank-one *subcomponents* $W_l \approx \sum_c \vec{U}^l_c (\vec{V}^l_c)^\top = U^l (V^l)^\top$, possibly overcomplete (more subcomponents than the matrix rank), to capture mechanisms in superposition.
- Subcomponents are trained to be parameter-faithful (sum to $\theta$), minimal, mechanistically faithful (robust to adversarially chosen ablations of unused subcomponents — the key difference from SPD's stochastic ablations), and simple.
- The paper builds attribution graphs between subcomponents ("circuits"), decomposes attention into cross-head subcomponents, and demonstrates model editing via parameter subcomponents.

## Artifact locations (from prev_paper.md appendix)

All model artifacts are on WandB (accessible via the logged-in `wandb` Python API):

Local layout: `models/<model>/target_model_<run>/` + `models/<model>/vpd_decomposition_<run>/` — each decomposition sits next to the model it decomposes.

**Main model (Pile, 4-layer), in `models/pile_4layer/`:**

- **Target model** — WandB run `goodfire/spd/t-9d2b8f02`. Files: `model_step_99999.safetensors` (268 MB; also a `.pt` on WandB), `model_config.yaml`, `final_config.yaml`, `tokenizer.json`.
  - Architecture (`model_config.yaml`): `LlamaSimpleMLP`, 4 layers, `n_embd=768`, 6 heads (GQA with 6 KV heads, i.e. effectively MHA), `n_intermediate=3072`, `n_ctx=512`, rotary (dim 128, base 10000), RMSNorm, no biases, vocab 50277 (GPT-NeoX/Pile tokenizer).
- **VPD decomposition (the subcomponents)** — WandB run `goodfire/spd/s-55ea3f9b`. Files: `model_400000.pth` (2.9 GB), `final_config.yaml`.

**SimpleStories model (2-layer), in `models/simplestories_2layer/`:**

- **Target model** — WandB run `goodfire/spd/gf6rbga0` (found via `pretrained_model_name` in the decomposition's `final_config.yaml`). Files: `model_step_99999.pt` (63 MB), configs, `tokenizer.json`.
  - Architecture: same family, 2 layers, `n_embd=192`, 6 heads (3 KV heads), `n_intermediate=768`, `n_ctx=512`, rotary dim 32, vocab 4019 (tokenizer `SimpleStories/test-SimpleStories-gpt2-1.25M`).
- **VPD decomposition** — WandB run `goodfire/spd/s-eab2ace8`. Files: `model_400000.pth` (98 MB), `final_config.yaml`. Subcomponent counts per matrix (from config): `mlp.c_fc` C=1152, `mlp.down_proj` C=960, `attn.q_proj`/`k_proj` C=288, `v_proj` C=384, `o_proj` C=480.
- **Other runs referenced in the paper** (not downloaded; see paper appendix ~line 2240): all decompose the same `t-9d2b8f02` target unless noted.
  - VPD multiseed: 5 seeds of the main decomposition (seed-consistency table).
  - Capacity sweep: 0.5×/1×/2×/4× subcomponent counts (feature-splitting analysis; 1× = main run).
  - No-adversarial-loss control `s-05ef623e`: same config minus the adversarial ablation loss (≈ SPD-style).
  - Hidden-act aux-loss VPD `s-aa4fec0a`: adds stochastic-forward-pass hidden-activation MSE loss.
  - PLT/CLT baselines: per-layer & cross-layer transcoders, BatchTopK k∈{8,16,32,64}, 4k/32k dicts, local-MSE and end-to-end-KL objectives (`mats-sprint/pile_*` WandB projects).
  - SimpleStories `s-eab2ace8`: a *different* 2-layer model (same architecture) trained on SimpleStories + its VPD decomposition; cleaner components, narrower distribution — good sandbox.
- **Code:** the paper's library is cloned at `param-decomp/` (https://github.com/goodfire-ai/param-decomp) — use it to load the checkpoints. Main package: `param_decomp/`; there is also `nano_param_decomp/`.

## Checkpoint file formats

All checkpoints are plain PyTorch **state dicts** (named tensors), not pickled model objects — instantiate the architecture from `param-decomp` (or read tensors directly).

- Target checkpoints: flat state dict of `LlamaSimpleMLP` (`wte.weight`, `h.<l>.attn.{q,k,v,o}_proj.weight`, `h.<l>.mlp.{c_fc,down_proj}.weight`, `h.<l>.rms_{1,2}.weight`, ...).
- Decomposition checkpoints (`model_400000.pth`) contain three key groups:
  1. `target_model.*` — frozen copy of the target weights (self-contained file).
  2. `_components.<module>.{U,V}` — the subcomponent factors; subcomponent $c$ is rank-one $\vec V_c \vec U_c^\top$ (column $c$ of $V$, row $c$ of $U$); $VU$ reconstructs the weight matrix. Shapes e.g. SimpleStories `h-0-mlp-c_fc`: V (192,1152), U (1152,768).
  3. `ci_fn.*` — the causal-importance function: a small 4-block transformer (d=512) whose output head emits one importance value per subcomponent (7104 total for SimpleStories). Used to decide which subcomponents are active per token.

Architecture comparison between the two models: see `architecture_comparison.md`.

## Loading the models

Use `load.py` at the project root — **standalone, runs on the user's default Python 3.11** (only needs torch/yaml/safetensors). `load_pile_4l()` / `load_simple_2l()` / `load(name)` return `(model, parameter_components, tokenizer)` (tokenizer is a transformers `PreTrainedTokenizerFast` from the local tokenizer.json; also available alone via `load_tokenizer(name)`):

- `model`: a `LlamaSimpleMLP` from `model_def.py` (a 3.11-compatible port of the paper repo's model file; verified **bit-exact** against the library forward pass on identical inputs).
- `parameter_components`: a `ParameterComponents` with `.components["h.0.mlp.c_fc"].V` (d_in, C) and `.U` (C, d_out) — subcomponent c is the rank-one matrix V[:,c] U[c,:], and W ≈ (V U)ᵀ in nn.Linear orientation (`.reconstruct(module)` does this; relative faithfulness error ~1e-3). CI-function weights kept raw in `.ci_fn_state_dict`.

**When the 3.13 venv is still needed:** the `param_decomp` library's `ComponentModel` machinery (masked/ablated forward passes, evaluating the causal-importance function, sampling) — it can't run on 3.11 (uses 3.12+ typing). For that, run with `param-decomp-vpd\.venv\Scripts\python.exe` (venv created with `uv sync --frozen --no-dev` in `param-decomp-vpd/`) and use `ComponentModel.from_pretrained(<local .pth path>)`; on first use it fetches target weights from WandB into `~/param_decomp_out/`.

**Repo folders:** `param-decomp/` = main branch (JAX rewrite); `param-decomp-vpd/` = git *worktree* of it at the `vpd-paper` release tag (the Torch code matching the checkpoints) — don't delete `param-decomp/`, its `.git` holds both.

**Decision (2026-08-24): use the Torch (`vpd-paper`) version for this project.** The JAX rewrite on `main` (`param-decomp/`) is a possible future option worth keeping in mind: `param_decomp/targets/llama_simple_mlp.py` reimplements the same target model on a shared GLU-transformer engine, loads the same `t-9d2b8f02` weights (converted to safetensors), and is pinned numerically equivalent to the Torch model by test fixtures (`targets/tests/simple_mlp_equivalence/`). Potentially useful for fast large-scale forward-pass sweeps on Modal GPUs. Caveat: its new training/checkpoint format differs, so it cannot load the paper's decomposition checkpoint directly. Also of independent value in `param-decomp/`: `docs/handbook.md` and `docs/skill.md` (methodology guides), `nano_param_decomp/` (compact Torch reference implementation).

**Numerical gotcha:** the models use NewGELU (tanh approximation, GPT-2 style), not exact GELU — max difference ≈ 5e-4 at |x| ≈ 2.7. Any reimplementation of the forward pass must use `F.gelu(x, approximate="tanh")` or the explicit tanh formula. See `plots/plot_activations.py`.

## Training data

- Pile 4-layer model: `danbraunai/pile-uncopyrighted-tok-shuffled` on HF (pre-tokenized shuffled `monology/pile-uncopyrighted`, GPT-NeoX tokenizer, 513-token rows with `<|endoftext|>` document boundaries mid-row).
- SimpleStories 2-layer model: `SimpleStories/SimpleStories` on HF (LLM-generated children's stories, `story` column + style/topic metadata columns).
- `load.py` also fetches sample rows from both via the public datasets-server REST API (no HF login). `pile_samples(n, offset)` / `simple_samples(n, offset)` return a list of 1-D int64 token-id tensors: Pile rows passed through as-is (already tokenized, exactly as in training), stories encoded with the simple_2l tokenizer without the appended [EOS]. Decode with `load_tokenizer(...).decode(ids.tolist())`.

## Alive components & harvest data

- Per-matrix alive-subcomponent counts (paper definition: mean CI > 1e-6 on data ≈ "non-negligible CI on ≥1 token per ~1M tokens"): tables in `architecture_comparison.md`. Computed (2026-08-25) via `ComponentModel` + `calc_causal_importances(...).lower_leaky` over 100 training sequences; reproduces the paper's per-layer table to ~1.5%. Pile: 9,837/38,912 alive (25%); SimpleStories: 6,499/7,104 (91%).
- The paper's full active-component list + LLM labels come from the `pd-harvest`/`pd-autointerp` pipeline (`harvest.db`: per-component `mean_ci` + activating examples; `interp.db`: Gemini labels). **Not on WandB and not in the repo** — only on Goodfire's cluster under `PARAM_DECOMP_OUT_DIR`. To get them: ask a collaborator, or rerun harvest (worker runs without SLURM: `python -m param_decomp.harvest.scripts.run_worker`; autointerp additionally needs an OpenRouter key).
- The WandB decomposition runs do log per-matrix mean L0 (`train/l0/*`, `eval/l0/*` in the run summary) — exact values matching the paper's L0 column.
- Gotcha: `pile_samples` rows are 513 tokens but n_ctx = 512 — truncate to 512 before a forward pass.

## Analysis code

- `circles.py` (adapted from the user's "Abs in Sup" project): probes local geometry of the residual stream — output displacement along random directions, and Frobenius norms of Jacobians over a random 2-plane grid. Key conventions:
  - Layer coordinates in halves: 0 = post-embedding, 0.5 = after layer-1 attention, 1 = after layer 1, ..., n_layer = after last layer (final RMSNorm applied only when out_layer = n_layer).
  - `layers_fn(net, in_layer, out_layer, sequence=...)` returns the residual-stream map; `sequence=False` treats vectors as length-1 sequences, `sequence=True` runs real causal attention. `text_h0(net, tokens, in_layer)` gives the stream for a tokenized text (token ids as list[int] or 1-D tensor — e.g. `tokenizer.encode(text)` or a `pile_samples`/`simple_samples` row; no tokenizer argument).
  - Defaults everywhere: `sequence=True, vary='last'` (perturb + measure at the last token position only; output side is always the last position in sequence mode).
  - `vary='last'` Jacobians use KV-cached suffix differentiation (prefix keys/values cached without autodiff, jacfwd on a single position) — ~30x faster than naive autodiff and seq-len-independent; verified vs full-sequence jacrev on both models.
  - `text_change(net, tokens, ...)` / `text_jacobian(net, tokens, ...)` are token-input wrappers around the two plot functions.
  - Known corners, all ruled acceptable by the user (2026-08-24) — do not re-flag: `in_layer=n_layer` mixes pre-norm h0 with a post-norm map (irrelevant: only in_layer < out_layer is of interest); each call draws a fresh random plane (intentional); `sequence=True` default silently reinterprets a hand-built (B, d) vector batch as one sequence (acceptable: the main workflow is `text_jacobian`/`text_change`, where h0 is built internally and always consistent); stale demo comment/labels (user: don't worry about it).
- `load_simple_2l()`'s tokenizer is wrapped (`NoSpecialTokens`) so encode does NOT append [EOS]; the bare `load_tokenizer("simple_2l")` still does.

## Project state

- **2026-08-24:** Project started. Located and downloaded the target models and VPD decompositions from WandB (see above); cloned param-decomp repo; built standalone 3.11 loading (`load.py`, `model_def.py`) and residual-stream geometry tooling (`circles.py`). Not yet a git repository. Next steps / specific research questions TBD with the user.

## Environment

- Windows 11, VS Code, PowerShell. Modal available for remote compute.
