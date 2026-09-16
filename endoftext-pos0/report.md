# `<|endoftext|>` and attention sinks in pile_4l

Session notes, 2026-08-28. Question that started it: **does the model architecture
have a built-in attention sink?** Answer: no — so we looked for an *emergent* one,
and found the classic "massive activations" signature on `<|endoftext|>` (EOS,
token id 0).

## 1. No built-in sink in the architecture

`CausalSelfAttention` in `prev_paper/model_def.py` is vanilla causal softmax
attention:

$$A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_\text{head}}} + M_\text{causal}\right)$$

so every query's attention weights sum to exactly 1 over real tokens. None of the
known built-in sink mechanisms are present:

- no learned per-head sink logit (à la GPT-OSS), and no "softmax-off-by-one"
  $\mathrm{softmax}_1(x)_i = e^{x_i}/(1+\sum_j e^{x_j})$ that would let a head
  attend to nothing;
- no dedicated sink/BOS token: training rows are 513-token Pile chunks with EOS
  only at document boundaries mid-row — there is no always-present position-0
  special token;
- no attention biases (no biases anywhere in the model), so heads can't even
  shift logits to soften the constraint.

Any sink behavior must therefore be emergent: softmax forces each head to put its
probability mass *somewhere*, and a low-information token is the natural dump.

## 2. Prior observation: EOS dominates the pre-`ln_f` norm tail

From the token-embeds final-activation cache
(`token-embeds/hide/cache/final_acts_pile_4l.npz`, 200 Pile rows × 512 =
102,400 positions): pre-`ln_f` residual norms have median 26.0 with a right tail
to 89.5, and the tail is EOS — all top-20 norms are EOS positions, 79 of the 83
EOS positions land in the top 100, and every EOS position has $\|h\| \ge 49$.
Outsize residual norms on a semantically empty token are the "massive
activations" signature that accompanies emergent attention sinks in the
literature; EOS is the model's only special token and the natural sink candidate.

## 3. Norms by stage: `resid_stage_norms.png`

`hide/resid_stage_norms.py` (~1 min local GPU) runs pile_4l on the first 1,000
cached Pile rows (`context-loss/hide/cache/pile_rows.pt`, truncated to
n_ctx = 512 → 512,000 positions) and histograms the residual-stream norm
$\|h\|$ at every stage — after embedding, after each attention sublayer, after
each MLP sublayer, after the final norm — split into four position categories
(probability densities, so tiny groups are comparable to the bulk):

- **other positions** (n = 510,665)
- **`<|endoftext|>` mid-sequence** (n = 335)
- **position 0** of the window (n = 999) — added after the §5 discovery
- **`<|endoftext|>` at position 0** (n = 1 in this sample; drawn as a vertical
  line at its value, excluded from y-axis scaling)

Each group's full min–max range is drawn as a horizontal line at the top of
the panel in the group's color, so outlier bars too short to see at density
scale are still visible. Y-limits are set by the first three groups' histograms.
A second version, `resid_stage_norms_zoom.png`, crops each panel's y-axis to
fit only the two *least-tall* histogram groups (the two tallest are clipped) —
useful where a spiky group towers over the others (e.g. the after-embedding
panel, where the 335 identical EOS values make one bar of density ~116).
simple_2l is skipped: its cached stories contain no [EOS] positions (the
loader encodes stories without appending it).

Median $\|h\|$ per stage:

| stage | other | EOS mid-seq | position 0 | EOS at pos 0 (n=1) |
|---|---|---|---|---|
| after embedding | 0.66 | 0.86 | 0.66 | 0.86 |
| after attention 1 | 1.00 | 0.94 | 1.35 | 1.20 |
| after MLP 1 | 5.66 | 10.20 | 7.52 | 10.60 |
| after attention 2 | 7.24 | 9.38 | 8.43 | 10.42 |
| **after MLP 2** | 6.94 | **128.0** | **220.9** | **251.8** |
| after attention 3 | 12.0 | 128.4 | 221.1 | 251.7 |
| after MLP 3 | 12.2 | 107.6 | 192.6 | 223.5 |
| after attention 4 | 21.7 | 37.0 | 20.1 | 50.3 |
| after MLP 4 (= pre-`ln_f`) | 26.0 | 87.1 | 19.3 | 49.2 |
| after final norm | 109.7 | 113.0 | 93.6 | 101.1 |

Reading: the anomaly is a **mid-network phenomenon** — created by MLPs 1–2
(essentially all of it by MLP 2 = `h.1.mlp`, which writes $\sim 30\times$ at
position 0 and $\sim 18\times$ at EOS), held through layer 3, actively removed
by the last layer's attention, and invisible after `ln_f` (it never reaches
the logits with outsize scale). This is exactly the trajectory reported for
massive activations / attention sinks in larger models. Details:

- Position 0 is distinguishable from stage one: already after attention 1 its
  median is 1.35 vs 1.00 (mechanically forced — at position 0 the attention
  output is exactly the token's own $W_O W_V$ value, softmax over one key).
- The mid-stage distributions are narrow (EOS $128 \pm \sim 10$, pos 0
  $221 \pm \sim 10$) — fixed, context-independent vectors rather than
  context-dependent computation.
- Attention 4 cancels position 0 *completely* (median 20.1 ≈ bulk 21.7; after
  MLP 4 it is even below bulk, 19.3 vs 26.0, and stays low after `ln_f`, 93.6
  vs 109.7), while EOS keeps an elevated remnant (87 pre-`ln_f`).
- The single EOS-at-position-0 behaves like a position-0 token mid-network
  (252 — the largest norm anywhere), i.e. the positional mechanism dominates
  the token-identity one; after attention 4 it sits between the two groups.
- Range lines reveal exceptions: at least one position-0 token *skips* the
  massive treatment entirely (pos-0 min 6.35 at MLP 2, inside the bulk), and a
  few "other" positions reach the massive band (max 228 at MLP 2 — probably
  positions 1–2, worth a check).

## 4. How many EOS per training chunk? `eos_per_chunk.png`

`hide/eos_per_chunk.py` counts EOS over all 4,000 cached training rows
(`context-loss/hide/cache/pile_rows.pt` — exact 513-token chunks from the
shuffled pre-tokenized training set; EOS appears only at document boundaries
mid-row, never as a chunk prefix):

| EOS count | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| chunks | 2,984 | 677 | 281 | 54 | 3 | 1 |

Mean 0.35 EOS/chunk; **74.6% of chunks contain no EOS at all** (consistent with
mean Pile doc length ≈ 1,450 tokens ≫ 513). So if EOS is a sink, the model
trains without one in three quarters of its windows — heads needing a dump
target must either have a fallback (e.g. position 0, or whatever token is
locally cheapest) or tolerate its absence. This makes follow-up 3 below the
interesting comparison, and it also means the sink interpretation can't be the
whole story of the massive EOS vector: the model builds it *whenever EOS
appears*, but most windows never exercise it.

## 5. The histogram outliers are position 0 — the real (bigger) sink site

The x-axes in `resid_stage_norms.png` span min→max of the pooled data, and
several panels extend far past the visible bars (e.g. after MLP 2 reaches
~250). `hide/stage_outliers.py` prints the top-5 norm positions per stage
(same 1,000 rows): **the far outliers are the position-0 tokens of every row,
regardless of which token sits there** — `'deep'`, `' come'`, `' translate'`,
`' hard'`, an EOS that happens to open a row… ordinary tokens, united only by
being first in the window:

- After MLP 2 → after MLP 3: position 0 sits at $\|h\| \approx 250$ (vs
  $\approx 128$ mid-sequence EOS, $\approx 7$ bulk). Exactly 1,000 such
  positions (one per row, 0.2%) — hence $p_{99.9} \approx 220$ while
  $p_{99} \approx 10$: a third, invisible mode in the density histograms.
- The buildup also starts earlier: position-0/1 tokens are already the top
  outliers after attention 1 and MLP 1.
- After attention 4 the giant norms are gone — attention 4 cancels position 0
  even more completely than EOS: the pre-`ln_f` (after MLP 4) top tail is back
  to mid-sequence EOS at ~89, with no position-0 excess.
- After the final norm the outliers are mundane (code/boilerplate tokens at
  124 vs median 110).

This answers follow-up 3 before we ran it: the model builds its
massive-activation sink at **position 0 unconditionally** — the one position
every query can always attend to under causal masking — i.e. the classic
StreamingLLM-style first-token sink, learned even though training chunks start
at arbitrary mid-document tokens. Mid-sequence EOS gets a second, weaker
(~half-norm) instance of the same treatment. So "EOS is the sink" was the
wrong first guess from §2: EOS is the *portable* document-boundary variant of
a primarily positional mechanism, and the EOS-free 75% of windows (§4) are no
puzzle at all.

## 6. The after-EOS (document-opener) frequency direction: `after_eos_direction.png`

How often does each token appear *right after* `<|endoftext|>` (= open a
document), and is that encoded in embedding space?

**Counts** (`hide/after_eos_counts.py` → `hide/cache/after_eos_counts.npz`):
scanned the first 8 train shards of the actual training set
(`danbraunai/pile-uncopyrighted-tok-shuffled`, ~204 MB each, kept in the HF
cache) — 1.95M rows ≈ **0.998B tokens**, giving 693,462 after-EOS samples
(within-row successors only) plus total token counts over the same data.
Result: only **8,243 distinct tokens (16.4% of the vocab) ever open a
document**, and the count grows logarithmically (4.7k → 8.2k from shard 1 to
8), so the requested "top 50% of tokens by this measure" (25,138) is
unreachable at any realistic data size — everything past rank ~8k is an
unrankable tie at zero. The fit set is therefore all 8,243 openers. Top
openers: `Q` (116k — StackExchange), `The` (31k), `1`, `\n`, `[`, `A`, `/*`,
`---`, `#`, `//`, `In`, `This`, `Introduction`, …

**Direction** (`hide/after_eos_direction.py` → unit vector
`hide/after_eos_direction.npy`): same method as
`token-embeds/hide/freq_direction.py` — ridge of $y = \log(\text{after-EOS
count})$ on the centered embeddings, $\lambda$ by 5-fold CV (picked
$\lambda = 0$), correlations cross-validated:

$$\rho_\text{CV} = +0.898, \qquad r_\text{CV}(\log \text{count}) = +0.916.$$

Document-opener frequency is as well linearly encoded as general frequency
(w_freq: 0.82/0.91) — and it is an **essentially independent axis**:

- On the fit set, $\rho(\log \text{after-EOS count}, \log \text{total count})
  = +0.09$ — being a frequent token and being a frequent opener are nearly
  unrelated.
- $\cos(w_\text{eos}, w_\text{freq}) = +0.21$; baselines along generic axes
  predict nothing (fit-set PC1: $\rho = -0.15$; $w_\text{freq}$: $+0.08$).
- Residualizing $y$ on log total count before fitting changes nothing:
  $\cos(w_\text{res}, w_\text{eos}) = 0.993$ with the same CV correlations —
  $w_\text{eos}$ is already the frequency-independent opener axis.

Caveat (same as for $w_\text{freq}$): the raw coefficient spectrum has a spike
on the last (bias) PCs — a low-variance artifact; the variance-weighted
contribution $|w \cdot \text{PC}_k| \cdot \mathrm{var}_k$ lives in the top
~20 PCs. Interpretation: the embedding carries a dedicated
"can-start-a-document" scale, presumably read by the layers that process the
post-EOS position — a natural companion to the §3 sink machinery, and a
candidate input direction for whatever writes the massive EOS vector.

### 6b. The covariance direction v_cov and the embedding PCs: `after_eos_vcov.png`

`hide/after_eos_vcov.py` → unit vector `hide/after_eos_vcov.npy`. Definition
(as in remove-feq-dir):

$$v_\text{cov} \propto X_c^\top y_c = \sum_i (x_i - \bar x)(y_i - \bar y),
\qquad y = \log(\text{after-EOS count}),$$

over the 8,243 openers. Decomposed over the project-standard embedding PCA
basis (frequent-token PCs, sign: frequent-mean ≥ 0). Results:

- $\cos(v_\text{cov}, \text{PC}_k)$: dominated by **PC1** ($-0.61$, i.e.
  toward the *frequent* side), then PC4 ($-0.28$), PC5 ($-0.26$), PC12
  ($-0.23$); top-10 PCs hold $\|{\cdot}\|= 0.80$ of the mass, top-100 give
  0.91 (random baseline per PC: $1/\sqrt{768} = 0.036$). So unlike general
  frequency (whose $v_\text{cov} \approx$ PC1, cos 0.97), the opener
  $v_\text{cov}$ is only partially PC1 and otherwise spread.
- The PC1 dominance is a variance artifact, not a strong correlation: since
  $v_\text{cov} \cdot \text{PC}_k \propto \sigma_k \rho_k$, the top-variance
  PC wins even with modest $\rho$. Per-PC rank correlations with $y$ are all
  weak — best PC12 ($-0.31$), PC5 ($-0.28$), PC4 ($-0.26$), PC734 ($+0.24$,
  a bias-side PC), PC1 only $-0.23$ — vs the full ridge direction's CV
  $\rho = 0.90$. **No single embedding PC correlates well with opener
  frequency; the signal is distributed**, same as for general frequency.
- $v_\text{cov}$ is also a much weaker predictor than the ridge direction
  (in-sample Spearman $+0.61$ vs $+0.90$ CV) and only half-aligned with it:
  $\cos(v_\text{cov}, w_\text{eos}) = 0.51$;
  $\cos(v_\text{cov}, v_\text{cov}^\text{freq}) = 0.57$,
  $\cos(v_\text{cov}, w_\text{freq}) = 0.16$.
- Figure bonus: in the projection scatter, the never-after-EOS tokens show
  the rare-token cone as a tight strip at projection $\approx -0.44$.

Note the apparent tension with §6 ($\rho(\log\text{after-EOS},
\log\text{total}) = 0.09$ yet $v_\text{cov}$ tilts strongly onto PC1):
covariance weights each PC by its variance, so a weak correlation along the
dominant-variance axis (PC1, $\rho = -0.23$) still contributes the largest
single component.

**Projecting out PC1 first** (checked 2026-08-28, quick computation — not in
the script): since projection commutes with covariance,
$v'_\text{cov} \propto (I - p_1 p_1^\top) X_c^\top y_c$ is exactly the old
$v_\text{cov}$ with its PC1 component deleted
($\cos(v', v) = \sqrt{1 - 0.61^2} = 0.792$, verified numerically to 1e-6).
It *improves* prediction: in-sample Spearman $0.606 \to 0.690$, Pearson
$0.618 \to 0.718$, and moves toward the ridge direction
($\cos(v', w_\text{eos}) = 0.58$ vs 0.51) — PC1's share was pure
variance-weighting artifact. This is one step of the $\Sigma^{-1}$ whitening
that turns $v_\text{cov}$ into $w$ ($w \propto \Sigma^{-1}\mathrm{Cov}$);
remaining top alignments are the old ones rescaled (PC4 $-0.35$, PC5 $-0.33$,
PC12 $-0.29$).

## 7. The sink test: attention mass + value output — `attention_mass.png`

Motivated by the (correct) objection that 74.6% of windows have no EOS, so EOS
can't be *the* sink, and that a sink needs **both** high attention mass **and**
a near-zero value contribution. `hide/attention_mass.py` (500 Pile rows)
recomputes the per-head attention matrices $A$ (the model runs flash
attention, so they're never materialized) and, per (layer, head): mass on key
0 (= first token of the *chunk*, arbitrary mid-document in ~75% of rows),
mass on mid-sequence EOS keys vs a position-matched baseline (same key
positions in EOS-free rows), the hand-off comparison, and the value-path
output norm $\|W_O^h v_p\| = \sqrt{v_p^\top W_O^{h\top} W_O^h v_p}$ (what key
$p$ writes into the stream if attended to).

**Mass on key 0** (queries ≥ 64; uniform ≈ 0.005):

| | h0 | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|---|
| L0, L1 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 | ~0.000 |
| L2 | 0.21 | 0.25 | 0.22 | 0.29 | **0.49** | 0.30 |
| L3 | 0.20 | **0.000** | 0.21 | 0.30 | 0.22 | 0.26 |

- **Layers 0–1 have no sink at all** (mass at or below uniform). The sink
  turns on exactly where the massive vectors exist (§3: built by MLP 2 =
  `h.1.mlp`) — layers 2–3 park 20–30% of *every* head's mass there, flat in
  query position (panel 2).
- **L2h4 (49%) is the position head from pile-qk-comps** — the head whose q/k
  subcomponents read the slow RoPE planes (near-linear coarse position).
  That's presumably *how* a position-based sink is implemented without any
  always-present token: it can find "position 0" positionally.
- **Value side: the sink signature checks out.** At the sink positions the
  value-path output is suppressed to 0.10–0.39× the all-position mean in L2
  (similar in L3) — the parked mass writes several-fold less than it would on
  normal keys. Not literally zero, but clearly attend-to-(almost)-nothing.
- **EOS is a *second*, functioning sink, not just "large for other reasons":**
  mid-sequence EOS keys attract 0.07–0.45 mass in the L2/L3 sink heads
  (10–100× the position-matched baseline; L2h4 again the largest at 0.45)
  with the same suppressed value output (~0.1–0.35×). And the **hand-off is
  real**: for queries with an EOS behind them, mass on key 0 drops to
  0.4–0.8× (L2 layer mean 0.31 → 0.18) — the EOS takes over part of the sink
  load. So: position 0 = primary, always-available sink; EOS = portable
  secondary sink at document boundaries.
- Layer 0's attention to EOS (h0: 0.12, ~15× baseline) has *normal-size*
  values (ratio 0.56–1.3) — that's functional document-boundary reading, not
  sinking, and it happens before the massive vector exists.
- **L3h1 anomaly, likely the cancellation mechanism:** it is the one sink-layer
  head with zero mass on key 0 from distant queries, yet its value output at
  position 0 is **24×** the mean (18× at EOS) — a huge write available
  exactly at the sink positions. The query at position 0 attends to key 0
  with mass 1 by construction (softmax over one key), so L3h1 dumps
  $W_O^{h1} v_0$ into position 0's own residual — presumably the attention-4
  cancellation of the massive component seen in §3. (Testable: sign of that
  write vs the massive vector, and $A[p,p]$ at EOS positions.)

## 8. Remaining follow-ups

1. Pairwise cosines of the massive residual vectors at after-MLP-2 (position-0
   group, and EOS group): is each a single fixed direction, and is it the
   *same* direction for both?
2. Confirm the L3h1 cancellation story: cosine of its position-0 write against
   the massive vector; its diagonal attention at EOS positions.
3. Does anything *read* the after-EOS direction $w_\text{eos}$ (§6) at
   post-EOS positions — e.g. the layers that predict document openers?
