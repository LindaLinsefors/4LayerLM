# What the 4-layer Pile model can and cannot do

*Capability analysis of the `pile_4l` target model (67M params, 4 layers, $n_{\text{embd}}=768$, trained on the Pile). 2026-08-26.*

## 1. Method

Three complementary passes, all in this folder:

1. **Quantitative** (`compute.py`, `metrics.py`, `pos_check.py`): the model was run over 400 held-out-style training rows (513 tokens each → $N = 204{,}800$ next-token predictions). For every prediction we recorded the loss $\ell_t = -\log p(x_{t+1} \mid x_{\le t})$, the rank of the true token, the full-distribution entropy, and the top-10 candidates. Automatic breakdowns: token class, copying conditions, calibration, error anatomy, frequency, context length.
2. **Targeted probes** (`probes.py`): hand-built prompts isolating single skills — induction, bracket matching, factual recall, grammatical minimal pairs, counting, arithmetic, idioms, code conventions, format induction. Each prompt is run as `<|endoftext|> + prompt`, matching training document starts.
3. **Qualitative** (`sample_for_review.py`, 8 LLM judges, `aggregate_reviews.py`): 339 stratified examples (near-misses, moderate errors, catastrophic errors with $\ell \ge 8$ nats, confident/hesitant correct predictions) were judged one by one: is the wrong guess plausible? which aspects (spacing, case, POS, topic, format) did it get right? what capability would the true token have required?

**Headline numbers:**

| metric | value |
|---|---|
| top-1 / top-5 / top-10 accuracy | 48.8% / 68.7% / 75.1% |
| mean loss / perplexity | 2.71 nats / 15.1 |
| median loss / median rank of true token | 1.71 nats / 2 |

So the *typical* token is nearly a coin-flip between the model's top two candidates, and the mean loss is dominated by a heavy tail of hard tokens (mean $\gg$ median).

## 2. When does it guess right?

### 2.1 By token class

| true-token class | share | top-1 | top-10 | mean loss |
|---|---|---|---|---|
| word with leading space | 48.2% | 32.6% | 63.8% | 3.73 |
| word continuation (subword) | 9.3% | 63.4% | 81.7% | 2.02 |
| punctuation | 20.6% | 69.4% | 94.1% | 1.25 |
| number | 5.5% | 41.3% | 67.7% | 2.94 |
| whitespace/newline | 6.2% | 84.1% | 97.4% | 0.69 |
| `<|endoftext|>` | 0.1% | 44.2% | 92.5% | 1.87 |

Essentially **all of the difficulty lives in choosing the next word**; the "glue" of text — punctuation, whitespace, finishing a word already begun — is largely solved. Notably the model detects document *endings* well above chance (44% top-1 on EOS), and the judges saw it confidently propose EOS after rhetorically conclusive sentences.

### 2.2 Copying is the single biggest driver of success

Define, for a prediction at position $t$ with context $x_{\le t}$ and target $x_{t+1}$:

$$\text{bigram}(t) = \mathbf{1}\!\left[\exists\, j \le t-1 :\; x_j = x_t \,\wedge\, x_{j+1} = x_{t+1}\right],\qquad \text{seen}(t) = \mathbf{1}\!\left[x_{t+1} \in \{x_1,\dots,x_t\}\right]$$

| condition | share of tokens | top-1 | mean loss |
|---|---|---|---|
| bigram seen in context | 26.0% | **85.8%** | 0.62 |
| token seen, bigram not | 30.2% | 52.2% | 2.08 |
| token novel in context | 43.9% | **24.5%** | 4.39 |

**45.7% of everything the model gets right is copy-completable from its own context window.** The pure form of this skill is very strong: on random word sequences repeated twice (`probes.py`), top-1 accuracy on the second copy is **84.8%** vs 0.0% on the first — a textbook induction-head signature, and numerically identical to the in-the-wild bigram-seen accuracy.

The judges' post-hoc attribution of *why* confident-correct predictions succeeded points the same way: copy-from-context 32%, format/structure 28%, formulaic n-gram 20%, syntax-forced 16% — and **semantic inference ≈ 0%**. When the model is right *and* sure, it is never because it understood something; it is because the answer was mechanically available (visible, formulaic, or grammatically forced). Genuine semantic inference appears only among *hesitant* correct predictions (13% of those).

### 2.3 Frequency and self-knowledge

Accuracy falls monotonically with token rarity (70.4% top-1 on tokens with >1k occurrences in the sample, 17.3% on hapaxes), and the model *knows* how hard each position is:

| top-1 confidence bin | mean conf | empirical acc |
|---|---|---|
| [0.9, 1.0) | 0.98 | 97.1% |
| [0.5, 0.6) | 0.55 | 53.9% |
| [0.1, 0.2) | 0.15 | 15.3% |

Expected calibration error is **0.007** — for practical purposes perfectly calibrated, with predictive entropy strongly tracking realized loss ($\rho = 0.66$; mean entropy 1.50 nats when right vs 3.92 when wrong). On genuinely arbitrary tokens (operands in generated math worksheets, arbitrary proper names) judges repeatedly noted near-flat top-5 distributions: the model correctly signals "this is not inferable". This calibration is itself a capability the original question list didn't include, and it is one of the model's most robust.

## 3. When it is wrong, how wrong is it?

**Mostly not very.** The 8 judges rated the model's wrong top-1 guesses (n = 259):

| plausibility of wrong guess | all errors | near-miss | moderate | catastrophic ($\ell\ge8$) |
|---|---|---|---|---|
| equally good as the truth | 35% | 63% | 27% | 27% |
| plausible | 41% | 23% | 50% | 37% |
| grammatical but contextually off | 19% | 8% | 22% | 25% |
| broken (violates grammar/format) | **5%** | 5% | 2% | 12% |

So roughly **three quarters of all "errors" are continuations a competent human writer could have produced**, and even among the catastrophic-loss tail, two thirds are. Loss conflates "model failed" with "text was genuinely unpredictable": judges repeatedly found huge-loss tokens that were corpus artifacts (scraping glitches, missing punctuation, idiosyncratic `....`), where the model's guess read *better* than the ground truth (e.g. loss 16.3 on a corrupted sentence where the model's ` delete` was the coherent choice).

### 3.1 Partial credit: what stays right when the word is wrong

Automatic string-level checks on all 104,898 errors, and spaCy POS tagging of the slot (substituting predicted vs true token into the context) on 2,829 of them:

| aspect preserved in error | rate |
|---|---|
| leading-space / word-boundary status | 84% |
| capitalization (both alphabetic) | 90% |
| token class (word/punct/number/…) | 77% |
| exact POS in slot (spaCy) | 39% |
| content-word vs function-word | 71% |
| true token still in top-5 / top-10 | 39% / 51% |

The judges' independent aspect labels agree (spacing 82%, topic 39%, format 38%, case 33%). Interpretation: **orthography and syntax-category are nearly always respected; what is missed is the specific lexical choice.** The model reliably knows *that* a capitalized proper noun, or a verb, or a closing bracket goes here — see also the minimal-pair results below — it just cannot always say *which one*. Two amusing micro-signatures: in 3.4% of errors the model predicts the token that comes *after* the true one (anticipation/skipping), and in 0.5% it re-emits the current token.

### 3.2 What would have been needed? (the actual failure taxonomy)

Judges assigned each error the capability that getting the true token would have required:

| requirement | share of errors |
|---|---|
| genuinely unpredictable (new name, arbitrary number, new info) | 32% |
| world knowledge the model lacks | 23% |
| local context (last ~20 tokens) it failed to use | 18% |
| answer literally visible in context, not copied | 8% |
| format/structure tracking | 7% |
| long-range context (>20 tokens back) | 5% |
| syntax | 5% |
| arithmetic / counting | 1% |

A third of the errors are not fixable by *any* capability — they are the irreducible entropy of text. The real capability gaps, in order: **missing world knowledge** (rare/technical vocabulary, entities), **failure to integrate nearby semantic constraints**, and — despite copying being its strongest skill — **unreliable copying** (§4.2).

## 4. Behavioral signatures (from 339 close readings)

These recurred independently across the 8 judges:

**4.1 The safe-generic reflex.** Where a specific rare noun is due, the model retreats to a high-frequency function word: `preservation of` (71.5%) where the text had `preservation from`; ` the` where `Agriculture` (named 30 tokens earlier) was due. The dominant bigram prior buries document-specific information — the flip side of the frequency table in §2.3.

**4.2 Copying is strong but *grabby*.** The induction machinery reaches for the most recent/salient matching string, not the semantically correct referent: completing `VIS`→`m` (medial) copied from earlier even though the sentence says "lateral" right there; 75% confidence on `roller` (from "roller-like member", 15 tokens back) where the patent meant the *photosensitive* drum; predicting `z`→`V` so as to spell the drug `zVAD-fmk` a second time in a slot where a *different* compound is due. And the converse failure: answers sitting in plain sight ~25–60 tokens back that were simply not reused (`dos`, `stream`, `Object`, a post's own title word). Copying at this scale is a *heuristic*, not reference tracking.

**4.3 Template-matching without topical grounding.** After "an indigenous pre-" in a text about Bronze-Age Anatolia, the model confidently produces `Columbian` — the globally most frequent filler of that construction, geographically absurd here. Similarly `Jewel` (photographer's first name in a photo credit) → `ry`, forming "Jewelry".

**4.4 Recently-reinforced local patterns over-generalize.** After three abbreviation definitions each ending in a period, it puts 77.6% on ending the fourth the same way (which continues instead). In enumerated lists it prefers *repeating the previous item* to advancing: `Monday, Tuesday,` → ` Tuesday` (0.152) with ` Wednesday` second (0.092); same for months; `- apples\n- oranges\n-` → ` or…` (re-starting " oranges"). Sequence *advancement* only wins when the pattern is overwhelmingly common (`1, 2, 3, 4,` → ` 5` at 0.78; `a, b, c,` → ` d` at 0.62).

**4.5 Semantics and format compete, and either can lose.** Sometimes format wins over meaning (a corporation in a legal party list gets `, Individually;` because four previous entries did). Sometimes meaning wins over format (excellent biological guesses in a slot where the document's own convention demanded a markdown `*` before the gene name — a pattern demonstrated twice in the visible window).

**4.6 Latent structure even in bad misses.** Guessing an astronomy journal volume it predicted 435/437/432/436 — the truth was 441: wrong token, correct numeric neighborhood and genre convention. Predicted "ages" after `Age:` in a key–value template were 30/31/33 — wrong digits, right *type*.

## 5. Probe scorecard

From `probes.py` (full numbers in `results/probes.json`):

**Solidly present**
- **Induction/copying**: 84.8% top-1 on the second copy of random 25-token sequences.
- **Subject–verb agreement, incl. across attractors**: 13/14 minimal pairs, usually by factors of 5–2000: `The keys to the cabinet` → ` are` beats ` is` 22×; `The key to the cabinets` → ` is` beats ` are` 4×; reflexive gender/number (`The girl hurt herself` 20×); past tense after "Yesterday" (877×).
- **Bracket/environment matching**: `f(x` → `)` 0.69; `[0, 1, 2` → `]` 0.63; LaTeX `\end{` after `\begin{equation}` → `equation` at **0.999**.
- **Idioms/collocations**: `Once upon a` → ` time` 0.73; `in order` → ` to` 0.85; `Ladies and` → ` gentlemen` 0.58; even `the quick brown fox … lazy` → ` dog` 0.21.
- **Code/markup conventions**: `import numpy as` → ` np` 0.82; `href="http` → `://` **0.99**; `if __name__ == "__` → `main` 0.88; and its "misses" are themselves idiomatic (`def main(` → `args`/`argv`; `#include <std` → `int` = `<stdint`; `SELECT * FROM` → `` ` `` = MySQL quoting).
- **Format induction**: Q:/A: alternation → `A` 0.61; key–value templates continued with the right value *type*.
- **Calibration** (§2.3).

**Partial / surface-level only**
- **Counting-like sequences**: `1,2,3,4,` → ` 5` works; weekdays/months lose to item-repetition (right answer rank 2); `2, 4, 6, 8,` → ` 9` narrowly beats ` 10` — memorized list order, no numerical pattern.
- **Facts — collocational, not relational.** Facts stored as strong surface associations work: `President Barack` → ` Obama` **0.97**, `deoxyribonucleic` → ` acid` 0.94, `Einstein developed the theory of` → ` relativity` (top-1), `William` → ` Shakespeare` (top-1). But any *relational lookup* fails: `The capital of France is` → ` Paris` at **rank 494**; `The chemical symbol for gold is` → ` Au` rank 462; `The Earth revolves around the` → ` Sun` rank 15. The knowledge base is a bigram association table, not a queryable relation store.
- **Quote closing** after a quoted clause: prefers continuing the sentence (arguably also plausible).

**Absent**
- **Arithmetic**: `2 + 2 =` → ` -` (0.34), with ` 4` at 0.049; `6 * 7 =` → ` 42` rank 48. Nothing resembling computation, only "an expression continues" statistics. (Consistently, only 1% of in-the-wild errors needed arithmetic — Pile text rarely does.)
- **Relational factual retrieval** (above).
- **Referent tracking / choosing the *right* antecedent to copy** (§4.2).
- **Non-English**: only 6 judged examples, but 50% of those errors were rated contextually off vs ~78% plausible-or-better for English domains.

## 6. Domain dependence

Share of errors rated plausible-or-better, by document domain (judged sample): math/formula 89%, news 85%, fiction 86%, legal boilerplate 79%, technical/scientific 78%, nonfiction prose 77%, web/forum 75%, code 64%, tables/lists 64%, non-English 50%. The model is most "reasonable" where text is formulaic (math notation, news register, legal boilerplate) and weakest where errors are objectively checkable (code identifiers, table consistency) or under-trained (non-English).

## 7. Summary picture

What four layers and 67M parameters buy, on the Pile:

$$\text{model} \approx \underbrace{\text{n-gram statistics}}_{\text{frequency, collocations, idioms}} + \underbrace{\text{induction/copy heads}}_{\text{85\% when applicable, but greedy}} + \underbrace{\text{local syntax engine}}_{\text{agreement, brackets, orthography}} + \underbrace{\text{format conventions}}_{\text{code, LaTeX, templates, EOS}} + \underbrace{\text{honest uncertainty}}_{\text{ECE } 0.007}$$

What is missing is everything that requires *binding*: attending to the semantically correct antecedent rather than the most salient one, querying a fact by relation rather than by surface collocation, overriding a global n-gram prior with document-specific information, and any computation (arithmetic, counting). Judged as a writer, the model almost never produces broken text (5% of errors) — it produces *defensibly generic* text, and its failures are failures of specificity, not of fluency.

## 8. Caveats

- "Catastrophic loss" ≠ "catastrophic failure": a sizeable slice of the high-loss tail is corpus noise, arbitrary tokens, or tokenization artifacts (e.g. ` size`+`able` for "sizeable", or "Gaz" parsed as a word-fragment) — trust the plausibility distributions over raw loss.
- Judges saw only the last 60 tokens of the 512-token context; the `bigram_seen_in_context` flag scans all 512, so it can be true without the pair being visible to the judge.
- Plausibility labels come from Sonnet-class LLM judges (single judgment each, no cross-validation); treat percentages as ±few %.
- Single-token probes under-credit multi-token answers; minimal pairs use joint probability over the candidate's tokens.
- All data is from the training distribution (400 rows ≈ 205k tokens of `pile-uncopyrighted-tok-shuffled`); this measures capability on-distribution, not generalization.

## 9. Files

| file | role |
|---|---|
| `compute.py` | forward pass → `results/predictions.npz` (~1 min GPU) |
| `metrics.py` | §2–3 automatic tables → `results/metrics.json` |
| `pos_check.py` | spaCy in-slot POS check → `results/pos_check.json` |
| `probes.py` | §5 probe suites → `results/probes.json` |
| `sample_for_review.py` | stratified 340 examples → `review/batches/` |
| `review/agent_out/` | per-example LLM judgments (8 × ~42) |
| `aggregate_reviews.py` | §3–4, §6 aggregation → `results/review_summary.json` |
