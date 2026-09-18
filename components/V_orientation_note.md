# The V-orientation bug (2026-09-17)

## Setup

VPD decomposes each weight matrix into rank-one subcomponents. For a decomposed
linear layer with input dimension $d_{\mathrm{in}}$ and $C$ subcomponents,
component $c$ is the rank-one matrix

$$W \approx \sum_{c=1}^{C} \vec V_c\, \vec U_c^{\top},$$

where $\vec V_c \in \mathbb{R}^{d_{\mathrm{in}}}$ is the component's **read-in
direction** and $\vec U_c \in \mathbb{R}^{d_{\mathrm{out}}}$ its write direction.
The layer's forward pass through the components is

$$y = \big((x V) \odot m\big)\, U, \qquad V \in \mathbb{R}^{d_{\mathrm{in}} \times C},\; U \in \mathbb{R}^{C \times d_{\mathrm{out}}},$$

with $m$ the CI mask. In this **library orientation**, $\vec V_c$ is **column**
$c$ of $V$: the component's input activation on token $t$ is

$$a_t \;=\; \vec V_c \cdot x_t, \qquad x_t = g \odot \frac{\mathrm{wte}[t]}{\mathrm{rms}(\mathrm{wte}[t])}.$$

This $a_t$ is the x-axis of every `v_dot_embeddings` / `v_dot_vs_ci` figure.

## The two layouts in our caches

| Source | Layout | Component $c$ is … |
|---|---|---|
| JAX `prepared_weights`, old `.pth` (`_components.*.V`, e.g. (768, 512)) | $V \in \mathbb{R}^{d_{\mathrm{in}} \times C}$ | column $V_{[:,c]}$ |
| uv dump caches `uv_{C,D,E,newA,newB}.npz` | $V^{\top} \in \mathbb{R}^{C \times d_{\mathrm{in}}}$ | **row** $V^{\top}_{[c,:]}$ |

The dump scripts saved $V^{\top}$ (one row per component — convenient for
subsetting to alive components), and the cos-sim read-in/write-out reports use
that layout correctly.

## The bug

The first `components/C` scripts assumed the *library* orientation for the
*dump* caches: they took

$$\vec v_{\mathrm{used}} = V^{\top}_{[:,\,c]} \quad\text{(a column of the row-major array)}$$

instead of

$$\vec v_{\mathrm{true}} = V^{\top}_{[c,\,:]}.$$

A column of $V^\top$ is the vector $\big( (V_1)_c, (V_2)_c, \dots, (V_C)_c \big)$ —
coordinate $c$ of *every* component's read-in — which has nothing to do with
component $c$. Geometrically it is close to a random direction.

**Why it wasn't caught:** for `h.0.attn.q_proj` and `k_proj` the matrix is
square, $d_{\mathrm{in}} = C = 768$, so both indexings return a length-768
vector and every downstream computation runs without error. For any MLP matrix
the shapes differ ($3072 \times 768$), and the code would have crashed
immediately — but only attention matrices were ever analyzed.

## The evidence

Test on C `h.0.attn.k_proj:513`, the crisp newline detector (92 % of its CI
mass on `'\n'`). Dot with the RMSNorm-ed embedding of `'\n'`:

| Vector used | $a_{\text{'\textbackslash n'}}$ | rank of $\lvert a\rvert$ among 50,277 tokens |
|---|---|---|
| row $V^{\top}_{[513,:]}$ (correct) | $+11.8$ | **1** |
| column $V^{\top}_{[:,513]}$ (bug) | $+0.53$ | 28,301 |

After the fix, every checked C detector has its trigger tokens as the top dots:

| Component | trigger | top dots |
|---|---|---|
| k:513 | newline | `'\n'` $+11.8$, `'\r\n'` $+10.0$, `'\n\n'` $+9.7$ |
| k:3 | period | `'.'` $+8.8$, `').'` $+4.9$ |
| k:259 | ' the' | `' the'` $+11.7$, `'the'` $+9.0$, `' The'` $+8.9$ |
| k:257 | digits | `'5'` $+12.3$, `'6'` $+12.2$, `' 4'` $+12.2$ |

## Consequences

- **Wrong and regenerated:** the $a_t$ axis of all C figures made before the
  fix (both figure types, q and k). The claimed finding that C's token
  detectors have unremarkable read-in dots — "token selectivity lives in the CI
  function, not in $\vec V_c \cdot x$" — was an artifact of the transposed
  read: a random direction produces exactly the "no correlation with CI"
  signature. Corrected C looks qualitatively like the old decomposition:
  **read-ins are embedding-aligned**.
- **Never affected:** all CI values and top-firing-token lists (computed from
  forward passes, never touching the cached $V$); the `components/Old` figures
  ($V$ loaded from the `.pth` in the unambiguous $(768, 512)$ library
  orientation); the cos-sim read-in/write-out reports (rows-as-components,
  matching the true dump layout).
- **Going forward:** `analyze_top20_new.py` (A/B/D/E) indexes rows, with an
  extra alive-set lookup for A/B whose dumps store only alive components.
