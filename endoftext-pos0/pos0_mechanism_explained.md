# The position-0 mechanism, explained (rendered-math version)

The two equation-heavy sections from the chat discussion (2026-08-31), verbatim
but with rendered equations. Context: [pos0_mechanism.md](pos0_mechanism.md).

## 0. What is position-blind, and where position can enter

With no positional embeddings, the stream entering layer 1 is
$h^{(0)}_p = W_E[t_p]$ — a pure function of the token. MLPs and RMSNorm act per
position, so they map a position-blind distribution to a position-blind
distribution. RoPE rotates $q$ and $k$ by the *relative* offset, so the
attention logit

$$\ell(p, j) = \frac{q_p^\top R_{p-j}\, k_j}{\sqrt{d}}$$

is a function of $(t_p, t_j, p-j)$ only — translation-invariant. If the context
extended infinitely to the left, every position would be statistically
identical, and no absolute position information would exist anywhere in the
network.

The **single** violation of translation invariance is the causal mask: the
softmax at query position $p$ runs over only $p+1$ keys,

$$a_{pj} = \frac{\exp\ell(p,j)}{\sum_{j' \le p} \exp\ell(p,j')},
\qquad
\mathrm{out}_p = \sum_h W_O^h \sum_{j \le p} a_{pj}\, v_j .$$

So *all* absolute-position information in this model is a boundary effect of
the softmax support, and "position $p$" is really "only offsets $\delta \le p$
exist behind me." At the extreme, $p = 0$ has one key, so $a_{00} = 1$ and

$$\mathrm{out}_0 = \sum_h W_O^h v_0 = W_O W_V \tilde h_0$$

exactly — a deterministic function of the first token (values are not
rotated). This is the closed form the graft experiment injects.

## 1. Why the truncated softmax shifts the *mean* output (the part a linear probe can read)

Truncation obviously changes the output *distribution* (fewer terms averaged →
higher variance, e.g. the norm bump 1.35 vs 1.00). But a linear probe reads
mean differences, so the interesting question is why
$\mathbb{E}[\mathrm{out} \mid p] \ne \mathbb{E}[\mathrm{out} \mid \text{bulk}]$.

Write the head's bulk behavior as an offset-resolved attended-value
decomposition: mass $\bar a(\delta)$ on offset $\delta$, with conditional mean
value $\bar v(\delta) = \mathbb{E}\!\left[W_{OV}\, x_{p-\delta} \mid \text{head
attends at } \delta\right]$. Then roughly

$$\mathbb{E}[\mathrm{out} \mid \text{bulk}] \approx \sum_\delta \bar a(\delta)\, \bar v(\delta),
\qquad
\mathbb{E}[\mathrm{out} \mid p] \approx \sum_{\delta \le p} \tilde a_p(\delta)\, \bar v(\delta),$$

where $\tilde a_p$ is $\bar a$ truncated to $\delta \le p$ and renormalized
(the mass that would have gone to missing offsets is redistributed over the
surviving keys in proportion to their logits). The mean difference is therefore

$$\Delta\mu(p) \;\approx\; \sum_{\delta > p} \bar a(\delta)\,
\bigl[\,\bar v(\le p,\ \text{renormalized}) - \bar v(\delta)\,\bigr].$$

Two things must both hold for this to be nonzero, and both do in layer 0:

- **$\bar a(\delta)$ is not flat.** Every layer-0 head is local: 0.10–0.29 of
  its mass on $\delta = 1$, 50–80% within $\delta \le 32$ (panel 1 of
  `pos0_attn1_mechanism.png`). So at small $p$ a large fraction of the head's
  usual mass is displaced.
- **$\bar v(\delta)$ depends on $\delta$.** Heads select keys by content
  jointly with offset — what a head typically attends to two tokens back (say,
  punctuation, newlines, determiners) has a different mean value than its own
  token. If $\bar v$ were offset-independent, truncation would only
  redistribute mass among identical means and $\Delta\mu$ would vanish; the
  probe would then find nothing despite the norm difference.

The measured $\Delta\mu(0)$ has norm $\approx 0.18$ (full layer; 0.04–0.09 per
head) against a residual norm of $\sim 1$ — small, and pointing substantially
along directions in which the bulk output has *low variance* (directions the
ordinary mixture rarely explores). That's why the raw mean-difference probe
gets only AUC 0.61 while the covariance-whitened LDA probe gets 0.82: the
signal is not big, it is *clean* — a modest shift along quiet axes. And it's
genuinely six small shifts, one per head, not one "position head": each head
alone probes at 0.51–0.61, jointly 0.75–0.82.

The same formula explains the smooth $p$-dependence: $\Delta\mu(p)$ shrinks as
the tail mass $\sum_{\delta > p} \bar a(\delta)$ shrinks, and the self-mass
curves (panel 2) show $\tilde a_p$ relaxing to $\bar a$ only around
$p \approx 30$–60. So layer 1 doesn't emit a "position-0 flag"; it emits a
graded "how truncated is my context" statistic. That's why position 1 sits at
AUC 0.93 too, and why position 1 sometimes crosses the downstream sink
threshold.

**Update 2026-08-31:** this mean-shift account is incomplete on its own.
Linda's variance/bimodality hypothesis was tested in two rounds and
**confirmed in essence**: layer 0 carries at least three mutually orthogonal
directions along which the pos-0 value mixture is genuinely bimodal (modes
near $\pm c$, no mean shift, held-out kurtosis ≈ 1.7 vs Gaussian-null 3.06),
readable only through an even nonlinearity — with the binary labels supplied
by token classes rather than random signs. The reliability of the final
decision comes from aggregating these with the (itself bimodal) mean-shift
channel across two layers. See
[pos0_variance_signal.md](pos0_variance_signal.md), Addenda 1–2.
