# pile-att-L0: q:28 × k-component attention scores vs distance

How much does the always-on query component `h.0.attn.q_proj:28` want to attend to what each
k component writes, as a function of query–key distance? Three heatmaps, one per section below;
each section explains exactly how its scores are computed.

Scripts in `hide/` (`qk28_scores_norm.py`, `qk28_scores_act.py`, `qk28_scores_act_heads.py`);
score matrices and activation statistics cached in `hide/cache/`.

## Setup common to all three maps

In layer-0 attention the pre-softmax score (logit) of head $h$ between query position $i$ and
key position $j \le i$ is

$$A^h_{ij} \;=\; \frac{\big(R_i\, q_i^h\big)\cdot\big(R_j\, k_j^h\big)}{\sqrt{d_h}},
\qquad d_h = 128,$$

with $q_i = W_Q\, x_i$, $k_j = W_K\, x_j$, $x = \mathrm{rms}_1(h_0)$, and $R_i$ the RoPE rotation
at position $i$. VPD writes $W_Q$ and $W_K$ as sums of rank-one subcomponents; component $c$'s
contribution to the query (resp. key) vector is

$$q_i^{(c)} = (V_c \cdot x_i)\, U_c \;=\; a_c(x_i)\, \hat u_c,
\qquad a_c(x) \equiv \lVert U_c\rVert\, (V_c \cdot x),$$

i.e. a **fixed direction** $\hat u_c = U_c/\lVert U_c\rVert \in \mathbb{R}^{768}$ (spanning all
6 heads, 128 dims each) scaled by a **signed, input-dependent activation** $a_c$. By bilinearity
the full logit decomposes over (q component, k component) pairs; every map here is the slab of
that decomposition belonging to q:28, against each of the **126 alive** k components
(harvest mean CI $> 10^{-6}$; rows in id order).

Two exact simplifications:

- **Only distance matters.** RoPE rotations satisfy $R_i^{\top} R_j = R_{j-i}$, so
  $(R_i u)\cdot(R_j v) = (R_D u)\cdot v$ with $D = i - j$. The maps therefore have axes
  (k component) × (distance $D = 0\ldots 511$), not $(i,j)$.
  Concretely $R_D$ rotates, within each head, the coordinate pair $(p,\,p{+}64)$ by
  $\theta_p = D / 10000^{p/64}$, $p = 0\ldots 63$ (the model's split-half pairing; wavelengths
  $\lambda_p = 2\pi \cdot 10000^{p/64} \approx 6.3$ to $6.3\times 10^4$ tokens).
- **Layer-0 attention input is token-only**: $x_i = \mathrm{rms}_1(\mathrm{wte}[t_i])$ depends
  only on the token id, so $a_c$ is a per-token-id scalar (no context dependence). (Causal
  importance is still context-dependent — the CI function reads context.)

## Map 1 — `qk28_scores_normU.png`: unit output vectors, no data

$$\mathrm{score}(c, D) \;=\; \sum_{h=0}^{5}
\frac{\big(R_D\, \hat u_{q28}^h\big)\cdot \hat u_c^h}{\sqrt{128}}$$

where $\hat u^h$ is the head-$h$ slice (dims $128h \ldots 128h{+}127$) of the full-768-dim
unit-normalized output vector, and the head contributions are summed. This is pure weight
geometry: how aligned q:28's output direction is with each k component's output direction after
rotating by $D$ tokens, at a standardized (unit) magnitude.

Caveats specific to this map:

- **Row signs are gauge.** A subcomponent is unchanged under $(V_c, U_c) \to (-V_c, -U_c)$, so
  the sign of each row is an artifact of the stored checkpoint; only $|{\cdot}|$ and the
  *shape* in $D$ are meaningful. (Map 2 fixes this.)
- The overall scale is arbitrary (unit vectors); the bound per head is $1/\sqrt{128} \approx 0.088$.
  Observed max $|\mathrm{score}| = 0.036$.

## Map 2 — `qk28_scores_act.png`: scaled by typical data activations

When q:28 is active on the query token and component $c$ is active on the key token, their exact
joint contribution to the logit is $a_{q28}(x_i)\, a_c(x_j)$ times the unit-vector score above.
Map 2 replaces the per-token activations by each component's *typical* value on tokens where it
matters:

$$\mathrm{score}(c, D) \;=\; s_{q28}\; s_c \sum_h
\frac{\big(R_D\, \hat u_{q28}^h\big)\cdot \hat u_c^h}{\sqrt{128}},
\qquad s_c \;=\; \big\langle\, a_c \,\big\rangle_{\text{CI}(c) > 0.1}.$$

The average is **signed** and runs over all tokens with $\mathrm{CI} > 0.1$ inside the
component's harvest.db activation examples (≤ 1000 windows of 41 tokens each, reservoir-sampled
from 20k×32 Pile sequences; between ~850 and ~24,500 qualifying tokens per component). Units are
now actual attention logits.

Checks done:

- **harvest's `component_activation` is exactly $a_c$**: recomputing
  $\lVert U_c\rVert\,(V_c \cdot \mathrm{rms}_1(\mathrm{wte}[t]))$ locally from the checkpoints
  reproduces the DB values to ~0.2% (three components spot-checked; possible because $a_c$ is
  token-only at L0).
- **The signed mean is not a cancellation artifact**: per component,
  median $|\langle a\rangle| / \langle |a|\rangle = 1.00$; only 3 of 126 k components fall below
  0.5 (mixed-sign activations), so $s_c$ is representative for essentially all rows.
- **Gauge invariance**: the product $s_{q28}\, s_c\, (\hat u_{q28}\!\cdot\!\hat u_c)$ is invariant
  under component sign flips — unlike map 1, these signs are physical.

Remaining approximations: $s_c$ collapses the within-component spread of $a_c$ across trigger
tokens to one number, and $\langle a_{q28} a_c \rangle$ is approximated by
$\langle a_{q28}\rangle \langle a_c\rangle$ (query- and key-token activations are on different
positions, so correlations enter only through token co-occurrence within a window). The map
describes the *pairwise* score contribution; the total logit also contains all other q-component
rows.

## Map 3 — `qk28_scores_act_heads.png`: map 2 split by head

Identical to map 2 except the head sum is not taken: panel $h$ shows

$$\mathrm{score}_h(c, D) \;=\; s_{q28}\; s_c\;
\frac{\big(R_D\, \hat u_{q28}^h\big)\cdot \hat u_c^h}{\sqrt{128}}.$$

Normalization is still over the full 768-dim vectors (not per head), so **the six panels sum
exactly to map 2** (asserted in the script); a component concentrated in one head shows its full
strength there and ~0 elsewhere. All panels share one color scale. This is the physically
relevant split: each head runs its own softmax, so it is the per-head logit that competes with
other keys.

## Findings (summary)

- The dense high-CI rows dominate, and all share one signature: **positive near, negative far**
  — k:309 (+14 at $D \le 4$ → −10 far), k:29 (+7 → −16), k:465, k:76, crossover at
  $D \approx 5$–30, reaching −23 logits. Since q:28 fires on 65% of query tokens and
  k:29/k:309 cover most key tokens, the q:28⟷dense-k pairing writes a distance-decaying
  logit profile onto nearly every query–key pair: a **locality/recency bias**, matching the
  observation that all six L0 heads are local.
- The only broadband-positive (far-attention-promoting) rows are weak: k:299 (punctuation /
  technical, ≈ +3), k:404 (≈ +2), k:371 (≈ +1).
- Per head, the locality profile is **cross-head**: k:29's near+/far− appears in all six heads
  (far −1.3 to −3.6 each; largest single-head cell anywhere is 6.1 logits vs 23 in the head
  sum). Head-specific exceptions: k:309's near-positive sits mostly in h4/h5 (+3.9/+2.9, and h4
  is the one head where k:309 does not suppress far keys), k:299's far boost is almost entirely
  h0, k:404's mostly h3.
- Distance dependence is smooth everywhere (drifts over hundreds of tokens, no fast
  oscillations) — consistent with L0's RoPE energy sitting in the long-wavelength band.
