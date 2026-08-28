# The frequency direction in token-embedding space, and how to remove it

*2026-08-27. Findings from one working session, spanning `token-embeds/freq_direction.py`, the reworked `simple-emb-clusters/` (old version archived in `archive/simple-emb-clusters-2026-08-27/`), and scratch computations reproduced by `neighbor_table.py` in this folder.*

**Starting point.** The PCA of the frequent/alive token embeddings (see `token-embeds/pca.py`) had shown that PC1 correlates with corpus frequency (Spearman 0.64 pile / 0.78 simple) and that the last PC(s) form an effective **bias direction**: an all-token shared offset with tiny variance, the no-bias network's substitute for a bias parameter. Questions of this session: is this known? what is the *best* frequency direction? which direction should be removed when comparing embeddings?

## 1. Prior work (literature check)

Both observations are established:

- **Top PCs ≈ frequency**: Mu & Viswanath, *All-but-the-Top* (ICLR 2018) — word embeddings have a large common mean vector plus dominant PCs encoding frequency, not semantics; subtracting the mean and top PCs improves semantic tasks. For transformers: Puccetti et al. (EMNLP Findings 2022) — BERT/RoBERTa "rogue/outlier dimensions" are driven by pre-training token frequency. Related: Gao et al. 2019 (representation degeneration: the anisotropy cone in tied-embedding LMs comes from the frequency-imbalanced softmax gradient — same mechanism as our rare-token cone).
- **Constant direction ≈ learned bias**: Sun et al. 2024, *Massive Activations in LLMs* — in bias-free LLaMA-family models a few input-independent huge activations act as fixed bias terms (perturbing breaks the model; fixing to their means does nothing), and **training with explicit bias parameters makes them disappear**. The common-mean-vector of Mu & Viswanath is the same object seen statically.

Our version is cleaner than most published ones (near-isotropic cloud, frequency measured directly), but the direction of every finding matches the literature; nothing was dis-confirmed.

## 2. Definitions

Per model, the fit set is the **frequent** (pile_4l: count ≥ 7 over the 4,000 cached Pile rows, $n = 24{,}437$, $d = 768$) or **alive** (simple_2l: count ≥ 10 over 20k stories, $n = 3{,}702$, $d = 192$) tokens. Let

$$X \in \mathbb{R}^{n \times d} \; \text{(raw embedding rows } e_i\text{)}, \qquad y_i = \ln(\mathrm{count}_i),$$
$$\tilde X = X - \mathbf{1}\bar x^\top, \qquad \tilde y = y - \bar y \mathbf 1, \qquad \hat\Sigma = \tfrac1n \tilde X^\top \tilde X = \textstyle\sum_k \mathrm{var}_k\, \mathrm{pc}_k \mathrm{pc}_k^\top.$$

Three candidate frequency directions (all unit-normalized, signs fixed so the projection correlates positively with frequency):

$$\mathbf{w} \;\propto\; \arg\min_w \|\tilde X w - \tilde y\|^2 + \lambda \|w\|^2 \;=\; (\tilde X^\top \tilde X + \lambda I)^{-1} \tilde X^\top \tilde y, \qquad \lambda = 0.01 \text{ (5-fold CV over } \{0, 10^{-2}, \dots, 10^3\}\text{)};$$

$$\mathbf{v}_{\mathrm{cov}} \;\propto\; \tilde X^\top \tilde y \;=\; \textstyle\sum_i (e_i - \bar x)(y_i - \bar y) \;=\; n \,\widehat{\mathrm{Cov}}(e, \ln \mathrm{count});$$

$$\mathbf{pc}_1 = \text{top right-singular vector of } \tilde X.$$

They are related by inverse-covariance reweighting:

$$w \;\propto\; \big(\hat\Sigma + \tfrac{\lambda}{n} I\big)^{-1} v_{\mathrm{cov}}, \qquad\text{i.e. componentwise}\qquad w \cdot \mathrm{pc}_k \;\propto\; \frac{v_{\mathrm{cov}} \cdot \mathrm{pc}_k}{\mathrm{var}_k + \lambda/n},$$

so $w \to v_{\mathrm{cov}}$ as $\lambda \to \infty$; $w$ maximizes $\mathrm{corr}(\tilde X w, \tilde y)$ (the multiple correlation $R$), $v_{\mathrm{cov}}$ maximizes $\mathrm{Cov}(\tilde X v, \tilde y)$ over unit $v$.

Throughout, $\rho$ = **Spearman rank correlation** (Pearson correlation of average ranks; invariant under monotone transforms, so $\rho(\cdot, \mathrm{count}) = \rho(\cdot, \ln \mathrm{count})$). "CV" values are computed on out-of-fold ridge predictions (5-fold), so they are not inflated by the $d$ fitted parameters. **Removing** a direction $v$ means $\tilde X^{\perp} = \tilde X (I - vv^\top)$; **residual decodability** after removal is the CV $\rho$ of a fresh full ridge fit on $\tilde X^{\perp}$ — i.e. how well an adversary who refits can still rank tokens by frequency.

## 3. The frequency direction is real, strong, and *distributed*

Fit results (`token-embeds/freq_direction.py` → `freq_direction.png`, unit vectors in `freq_direction_{pile_4l,simple_2l}.npy`):

| | pile_4l | simple_2l |
|---|---|---|
| CV $\rho(\tilde Xw, \mathrm{count})$ | **+0.82** | **+0.88** |
| $\;$ (PC1 alone, for comparison) | 0.64 | 0.78 |
| CV Pearson $r(\tilde Xw, \ln \mathrm{count})$ | +0.91 | +0.91 |
| $\cos(w, \mathrm{pc}_1)$ | 0.25 | 0.39 |

The encoding is **distributed**, not concentrated in the top PCs — CV $\rho$ of ridge fits restricted to PC subsets:

| PC subset | pile_4l | simple_2l |
|---|---|---|
| PC1 only | 0.64 | 0.78 |
| top 10 | 0.74 | 0.81 |
| top 100 | 0.76 | 0.85 |
| last 10 only | 0.13 | 0.11 |
| all but last 10 | 0.80 | 0.86 |
| all $d$ | **0.82** | **0.88** |

Two caveats established along the way: (i) $w$'s visually largest coefficient sits on the **last (bias) PC** ($|w \cdot \mathrm{pc}_{\text{last}}| \approx 0.5$ in both models) — a pure $1/(\mathrm{var}_k + \lambda/n)$ artifact; those PCs contribute almost nothing to prediction (see table) and their variance-weighted contribution $|w\cdot\mathrm{pc}_k| \cdot \mathrm{var}_k$ is invisible next to PC1's. (ii) The counts themselves carry Poisson noise (pile: only 2.05M tokens, so $\mathrm{sd}(\ln \mathrm{count}) \approx 0.4$ near the cutoff) — the correlations above are lower bounds on how linearly decodable true frequency is. Out of the fit set, unfrequent tokens continue the count-vs-projection trend monotonically; the 304 missing tokens sit far below it at the push-away-cone value.

## 4. Which direction to remove: $v_{\mathrm{cov}}$ (≈ PC1), **not** $w$

Head-to-head on the fit sets (bias PCs = last 2 for pile, last 1 for simple):

| model | direction | $\rho(\tilde Xv, \mathrm{count})$ | mass in bias PC(s) $\|P_{\text{bias}} v\|$ | % emb-variance removed | residual decodability after removal |
|---|---|---|---|---|---|
| pile_4l | $\mathrm{pc}_1$ | +0.64 | 0.000 | 2.7% | $+0.41$ |
| pile_4l | $w$ | +0.83 | **0.567** | 0.3% | $+0.78$ |
| pile_4l | $v_{\mathrm{cov}}$ | +0.68 | 0.006 | 2.5% | $|{-0.27}|$ |
| simple_2l | $\mathrm{pc}_1$ | +0.78 | 0.000 | 7.0% | $+0.28$ |
| simple_2l | $w$ | +0.89 | **0.572** | 1.4% | $+0.82$ |
| simple_2l | $v_{\mathrm{cov}}$ | +0.80 | 0.012 | 6.8% | $|{-0.34}|$ |

- **$w$ is the best predictor and the worst removal direction.** 57% of its mass lies in the bias PC(s) (the suspected spurious bias-direction correlation, confirmed — it's the $\hat\Sigma^{-1}$ amplification of near-zero-variance directions), and removing it leaves frequency essentially fully decodable: best-to-predict $\neq$ best-to-remove.
- **$v_{\mathrm{cov}}$ is the principled removal direction**: for every remaining direction $a$, $\mathrm{Cov}(\tilde X^{\perp} a, \tilde y) = a^\top (I - \hat v \hat v^\top)\, \tilde X^\top \tilde y = 0$ — one projection zeroes *all* linear covariance with $\ln \mathrm{count}$. What remains ($|\rho| \approx 0.3$, negative sign not meaningful) is nonlinear remnant, not a frequency axis.
- **$\cos(\mathrm{pc}_1, v_{\mathrm{cov}}) = 0.965$ (pile) / 0.983 (simple)** — the dominant variance direction essentially *is* the frequency-covariance direction, so plain PC1 removal after centering was already nearly optimal; their residual decodabilities straddle each other across the two models.
- No single-direction removal can delete the distributed tail (§3); expect residual $|\rho| \approx 0.3$–0.4 either way.

**Recipe** for comparing token embeddings: restrict to frequent/alive tokens, **mean-center** (this alone kills the bias offset and the anisotropy — see §5), then project out $v_{\mathrm{cov}}$ or $\mathrm{pc}_1$. This is exactly Mu & Viswanath's all-but-the-top, with $v_{\mathrm{cov}}$ as a principled replacement for "top PCs".

## 5. Application: reworked simple_2l embedding maps (`simple-emb-clusters/`)

The whole t-SNE + clustering analysis was rerun on the 3,702 alive tokens with $\tilde A(I - vv^\top)$, both variants ($v = \mathrm{pc}_1$ and $v = v_{\mathrm{cov}}$) side by side. Findings:

- **The variant choice is irrelevant**: mean 10-NN cosine-neighbor overlap between the two spaces is **0.94** (100% of tokens share ≥ 5/10 neighbors); same-method ARI between variants (0.43–0.81) *exceeds* the between-method ARI within one variant (0.23–0.53) — switching removal direction perturbs partitions less than switching clustering algorithm. Only asymmetry: linear frequency decodability after removal is $+0.28$ (pc1) vs none ($|{-0.34}|$ nonlinear, vcov), as §4 predicts.
- **Anisotropy of the alive cloud was entirely the shared mean**: mean pairwise cosine 0.26 (raw alive rows) → −0.0002 after centering alone.
- **De-frequencying costs nothing semantically and removes the frequency organization**: the old dominant frequency banding of the t-SNE map is gone; every POS × semantic neighborhood survives. The old headline KMeans k=2 frequency split disappears — max silhouette drops 0.082 → 0.027, so "continuum, not clusters" gets *stronger*. Silhouette-optimal partitions are now semantic, not frequency tiers.
- **New discrete structure resolved**: HDBSCAN in the native 192-d space (previously 0 clusters / 100% noise) now isolates a crisp locative-preposition cluster — `in through around under behind above among beneath across within against below between …` (n ≈ 15, identical in both variants) — plus a broad high-frequency core.

## 6. Nearest-neighbor comparison: raw vs PC1-removed vs $v_{\mathrm{cov}}$-removed

Generated by `neighbor_table.py` (seed 0; queries sampled uniformly from the frequent/alive set, neighbors ranked by cosine within the same set; raw = uncentered rows, removed variants = centered + projected). What it shows: for pile the neighbor lists barely change (they were already frequency-matched morphological/semantic families); for simple_2l the removal visibly cleans up rarer queries — fragment junk neighbors (`b`, `cand`, `cann` for `jelly`; fragment soup for `glanc`) are replaced by semantic neighbors (`candy`, `ice`; `looks`, `gaze`, `staring`). The pc1 and vcov columns differ only in rank order, almost never in membership — §5's conclusion, visible token by token.

Legend: `_` marks a leading space in pile tokens (GPT-NeoX tokenizer); `##` marks SimpleStories continuation pieces.

### pile_4l  (24,437 frequent tokens; neighbors drawn from the same set)

| token | top-10 (raw embeddings) | top-10 (PC1 removed) | top-10 (v_cov removed) |
|---|---|---|---|
| `_interpreting` | `_interpret`<br>`_interpretations`<br>`_interpretation`<br>`_interpreted`<br>`_analyzing`<br>`interpret`<br>`_examining`<br>`_evaluating`<br>`_discussing`<br>`_assessing` | `_interpret`<br>`_interpretation`<br>`_interpretations`<br>`_interpreted`<br>`_analyzing`<br>`interpret`<br>`_examining`<br>`_evaluating`<br>`_discussing`<br>`_calculating` | `_interpret`<br>`_interpretation`<br>`_interpretations`<br>`_interpreted`<br>`_analyzing`<br>`interpret`<br>`_examining`<br>`_evaluating`<br>`_discussing`<br>`_calculating` |
| `foreach` | `forEach`<br>`while`<br>`for`<br>`each`<br>`Each`<br>`catch`<br>`++)`<br>`elif`<br>`isset`<br>`_Each` | `for`<br>`while`<br>`forEach`<br>`each`<br>`if`<br>`using`<br>`Each`<br>`For`<br>`_For`<br>`catch` | `for`<br>`while`<br>`forEach`<br>`each`<br>`if`<br>`using`<br>`Each`<br>`catch`<br>`For`<br>`try` |
| `_hey` | `_hello`<br>`_hi`<br>`Hey`<br>`_yeah`<br>`_Hi`<br>`_alright`<br>`Hello`<br>`_okay`<br>`Hi`<br>`ր` | `_hi`<br>`_hello`<br>`_Hi`<br>`Hey`<br>`_yeah`<br>`Hi`<br>`Hello`<br>`_oh`<br>`_okay`<br>`_alright` | `_hi`<br>`_hello`<br>`_Hi`<br>`Hey`<br>`_yeah`<br>`Hi`<br>`Hello`<br>`_okay`<br>`_oh`<br>`_alright` |
| `_cameras` | `_camera`<br>`_photography`<br>`_sensors`<br>`_photographs`<br>`_photographer`<br>`_detectors`<br>`_antennas`<br>`_phones`<br>`_photographic`<br>`_lasers` | `_camera`<br>`_photography`<br>`_sensors`<br>`_photographs`<br>`_detectors`<br>`_images`<br>`_phones`<br>`_photographer`<br>`_lenses`<br>`_imaging` | `_camera`<br>`_photography`<br>`_sensors`<br>`_photographs`<br>`_detectors`<br>`_images`<br>`_phones`<br>`_photographer`<br>`_lenses`<br>`_imaging` |
| `_plot` | `_plots`<br>`_Plot`<br>`plot`<br>`_plotted`<br>`Plot`<br>`_graphs`<br>`_curves`<br>`_curve`<br>`_histograms`<br>`_graph` | `_plots`<br>`_Plot`<br>`plot`<br>`_plotted`<br>`Plot`<br>`_graph`<br>`_curve`<br>`_curves`<br>`_graphs`<br>`_chart` | `_plots`<br>`_Plot`<br>`plot`<br>`_plotted`<br>`Plot`<br>`_graph`<br>`_curve`<br>`_curves`<br>`_graphs`<br>`_chart` |
| `25` | `24`<br>`26`<br>`30`<br>`35`<br>`20`<br>`27`<br>`15`<br>`23`<br>`22`<br>`28` | `24`<br>`26`<br>`30`<br>`35`<br>`20`<br>`15`<br>`27`<br>`23`<br>`22`<br>`28` | `24`<br>`26`<br>`30`<br>`35`<br>`20`<br>`15`<br>`27`<br>`23`<br>`22`<br>`28` |
| `ack` | `acks`<br>`acked`<br>`ACK`<br>`acking`<br>`acker`<br>`ick`<br>`ak`<br>`_Mack`<br>`ots`<br>`_hack` | `acks`<br>`acked`<br>`ACK`<br>`acking`<br>`acker`<br>`ak`<br>`ick`<br>`ac`<br>`ach`<br>`ok` | `acks`<br>`acked`<br>`ACK`<br>`acking`<br>`acker`<br>`ak`<br>`ick`<br>`ac`<br>`ach`<br>`ok` |
| `kes` | `KES`<br>`ke`<br>`ked`<br>`king`<br>`ken`<br>`zzles`<br>`KE`<br>`ker`<br>`_marriages`<br>`ks` | `ke`<br>`ks`<br>`king`<br>`KES`<br>`ked`<br>`ker`<br>`ken`<br>`KE`<br>`ket`<br>`ges` | `ke`<br>`ks`<br>`king`<br>`KES`<br>`ked`<br>`ker`<br>`ken`<br>`KE`<br>`ket`<br>`ges` |
| `_rules` | `_rule`<br>`_Rules`<br>`rules`<br>`Rules`<br>`_regulations`<br>`_guidelines`<br>`_laws`<br>`Rule`<br>`_Rule`<br>`_criteria` | `_rule`<br>`_Rules`<br>`rules`<br>`Rules`<br>`_regulations`<br>`_guidelines`<br>`_laws`<br>`Rule`<br>`_Rule`<br>`_criteria` | `_rule`<br>`_Rules`<br>`rules`<br>`Rules`<br>`_regulations`<br>`_laws`<br>`_guidelines`<br>`Rule`<br>`_Rule`<br>`_criteria` |
| `_true` | `true`<br>`_True`<br>`_false`<br>`_TRUE`<br>`false`<br>`True`<br>`_False`<br>`False`<br>`_truth`<br>`_FALSE` | `true`<br>`_True`<br>`_false`<br>`_TRUE`<br>`false`<br>`True`<br>`_False`<br>`False`<br>`_FALSE`<br>`_truth` | `true`<br>`_True`<br>`_false`<br>`_TRUE`<br>`false`<br>`True`<br>`_False`<br>`False`<br>`_FALSE`<br>`_truth` |

### simple_2l  (3,702 alive tokens; neighbors drawn from the same set)

| token | top-10 (raw embeddings) | top-10 (PC1 removed) | top-10 (v_cov removed) |
|---|---|---|---|
| `jelly` | `seaweed`<br>`chocolate`<br>`dolphins`<br>`coral`<br>`bubbles`<br>`pizza`<br>`cand`<br>`b`<br>`rainbows`<br>`cann` | `chocolate`<br>`candy`<br>`bubbles`<br>`coral`<br>`seaweed`<br>`ice`<br>`dolphins`<br>`pearl`<br>`bubble`<br>`rainbows` | `chocolate`<br>`candy`<br>`bubbles`<br>`coral`<br>`seaweed`<br>`ice`<br>`dolphins`<br>`pearl`<br>`bubble`<br>`rainbows` |
| `rules` | `puzzles`<br>`problems`<br>`moves`<br>`troubles`<br>`jokes`<br>`plans`<br>`traditions`<br>`list`<br>`##ions`<br>`lines` | `plans`<br>`moves`<br>`lines`<br>`games`<br>`puzzles`<br>`problems`<br>`jokes`<br>`riddles`<br>`traditions`<br>`paths` | `plans`<br>`moves`<br>`lines`<br>`games`<br>`puzzles`<br>`problems`<br>`jokes`<br>`riddles`<br>`traditions`<br>`paths` |
| `glanc` | `glanced`<br>`piec`<br>`gazing`<br>`chuck`<br>`peered`<br>`scrat`<br>`smi`<br>`snowf`<br>`staring`<br>`vent` | `looks`<br>`glanced`<br>`looking`<br>`gaze`<br>`pee`<br>`staring`<br>`shiver`<br>`gazing`<br>`frown`<br>`gazed` | `looks`<br>`looking`<br>`glanced`<br>`gaze`<br>`pee`<br>`staring`<br>`shiver`<br>`gazed`<br>`frown`<br>`gazing` |
| `balloon` | `kite`<br>`balloons`<br>`kites`<br>`bicycle`<br>`lantern`<br>`feather`<br>`rocket`<br>`fireworks`<br>`bubble`<br>`bike` | `kite`<br>`balloons`<br>`boat`<br>`feather`<br>`lantern`<br>`kites`<br>`rocket`<br>`cloud`<br>`bicycle`<br>`bike` | `kite`<br>`balloons`<br>`boat`<br>`feather`<br>`lantern`<br>`rocket`<br>`kites`<br>`cloud`<br>`bubble`<br>`bike` |
| `bloom` | `blossom`<br>`bloomed`<br>`blooms`<br>`blooming`<br>`blossomed`<br>`grow`<br>`thri`<br>`grows`<br>`flourished`<br>`sprou` | `blossom`<br>`bloomed`<br>`grow`<br>`blooms`<br>`blooming`<br>`flower`<br>`shine`<br>`blossomed`<br>`plant`<br>`twinkle` | `blossom`<br>`bloomed`<br>`grow`<br>`blooms`<br>`blooming`<br>`flower`<br>`shine`<br>`blossomed`<br>`plant`<br>`twinkle` |
| `##pp` | `b`<br>`ow`<br>`tapest`<br>`##ises`<br>`tim`<br>`snowf`<br>`empt`<br>`cand`<br>`##zz`<br>`riverb` | `ru`<br>`##ff`<br>`##zz`<br>`shape`<br>`##bb`<br>`##ms`<br>`##pping`<br>`ag`<br>`##ks`<br>`##sp` | `ru`<br>`##ff`<br>`##zz`<br>`shape`<br>`##bb`<br>`##ms`<br>`##ks`<br>`##pping`<br>`ag`<br>`lines` |
| `##j` | `fier`<br>`##in`<br>`##lec`<br>`##ank`<br>`##ject`<br>`##ud`<br>`glanced`<br>`3`<br>`cal`<br>`##read` | `##in`<br>`##v`<br>`##c`<br>`##x`<br>`emb`<br>`##re`<br>`##n`<br>`pl`<br>`fo`<br>`fier` | `##in`<br>`##v`<br>`##c`<br>`##x`<br>`##n`<br>`##re`<br>`emb`<br>`pl`<br>`fier`<br>`fo` |
| `brightly` | `brighter`<br>`vibrant`<br>`brightest`<br>`gracefully`<br>`lighting`<br>`faint`<br>`joyfully`<br>`freely`<br>`brightened`<br>`proudly` | `bright`<br>`brighter`<br>`vibrant`<br>`beautiful`<br>`softly`<br>`brightest`<br>`faint`<br>`gracefully`<br>`lighting`<br>`gently` | `bright`<br>`brighter`<br>`vibrant`<br>`brightest`<br>`softly`<br>`beautiful`<br>`faint`<br>`gracefully`<br>`lighting`<br>`gently` |
| `village` | `town`<br>`kingdom`<br>`neighborhood`<br>`forest`<br>`valley`<br>`towns`<br>`villagers`<br>`community`<br>`city`<br>`townspeople` | `town`<br>`kingdom`<br>`neighborhood`<br>`towns`<br>`forest`<br>`valley`<br>`villagers`<br>`city`<br>`community`<br>`townspeople` | `town`<br>`kingdom`<br>`neighborhood`<br>`towns`<br>`forest`<br>`valley`<br>`city`<br>`villagers`<br>`community`<br>`house` |
| `##ace` | `##itude`<br>`##lec`<br>`##ices`<br>`##iety`<br>`##here`<br>`##ance`<br>`##ison`<br>`##ushing`<br>`##aced`<br>`ow` | `##itude`<br>`##ance`<br>`##by`<br>`##ire`<br>`living`<br>`##ery`<br>`up`<br>`##ar`<br>`healing`<br>`##ment` | `##itude`<br>`##ance`<br>`##by`<br>`##ery`<br>`##ire`<br>`##ment`<br>`##ar`<br>`healing`<br>`##old`<br>`##lec` |

## Files

| file | contents |
|---|---|
| `report.md` | this report |
| `neighbor_table.py` | generates the table above → `neighbor_table.md` (change `N_QUERIES`/seed to resample) |
| `neighbor_table.md` | the generated table on its own |

Sources for §1: [All-but-the-Top](https://arxiv.org/abs/1702.01417) · [Outlier Dimensions Driven by Frequency](https://aclanthology.org/2022.findings-emnlp.93/) · [Massive Activations in LLMs](https://arxiv.org/abs/2402.17762) · [Representation Degeneration](https://arxiv.org/abs/1907.12009)
