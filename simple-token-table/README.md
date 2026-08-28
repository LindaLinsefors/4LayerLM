# SimpleStories token table

A classification of all **4,019 tokens** of the SimpleStories WordPiece tokenizer
(the tokenizer of the 2-layer model / VPD decomposition `s-eab2ace8`).

- `token_table.csv` — one row per token, ordered by token id (0–4018).
- `sorted_token_table.csv` — identical rows, sorted by `class` (in the order
  word, prefix, suffix, punctuation, digit, special), then `top_pos`
  (alphabetical), then `freq` (descending).
- `token_table.py` — rebuilds both (~5 min; needs `spacy` + `en_core_web_sm`,
  downloads a 17 MB story shard into `data/` on first run).

## How it was built

20,000 raw stories (the dataset's `test` parquet shard — i.i.d. with the training
split, and *raw* means capitalization is intact, which the tokenizer otherwise
destroys) were processed two ways:

1. **Tokenized** with the model's tokenizer exactly as in training → exact token
   frequencies (`freq`).
2. **POS-tagged** with spaCy (`en_core_web_sm`) on the raw text. Each alphabetic
   word occurrence is lowercased and WordPiece-tokenized: a word that maps to a
   single token contributes its POS tag to that token; a word split into several
   pieces contributes its POS to *every* piece. So for a whole-word token the POS
   distribution describes its own usage, while for a piece like `##ing` it
   describes the words it appears in.

## Columns

| column | meaning |
|---|---|
| `id` | Token id (row index into the model's embedding matrix). |
| `token` | The token string. `##x` means a continuation piece: glued to the previous token with no space (WordPiece convention). The `##` never appears in actual text. |
| `class` | Structural class, mutually exclusive — see below. |
| `freq` | Number of occurrences of this token when tokenizing the 20k-story sample (all counts in the table are from this sample). |
| `standalone_count` | Number of occurrences as a *complete word* (the word maps to exactly this one token). Compare with `freq`: for `to`, 102,131 of 102,483 occurrences are the word "to"; the rest are first pieces of split words like "tongue". |
| `top_pos` | The most common spaCy POS tag for this token (argmax of the `pos_*` columns). Empty for punctuation/digit/special tokens. |
| `name_frac` | **Fraction of proper-name usage.** See definition below. Empty when there are fewer than 3 mid-sentence occurrences to judge from. |
| `cap_frac` | Fraction of mid-sentence standalone occurrences that are capitalized, regardless of POS. Diverges from `name_frac` only for capitalized non-names — most notably `i` (cap_frac 1.0, name_frac 0.0). |
| `pos_NOUN` … `pos_INTJ`, `pos_other` | The POS distribution: fractions summing to ≈ 1 over the token's occurrences (for word tokens) or over occurrences of words containing it (for prefix/suffix tokens). Kept as a distribution because usage overlaps — e.g. `play` is 0.91 VERB, 0.09 NOUN. |
| `example_words` | For prefix/suffix tokens: the 3 most common words this piece occurs in (e.g. `##ness` → "vastness stillness forgiveness"). Empty for tokens that always stand alone. |

## `class` values

| class | count | meaning |
|---|---|---|
| `word` | 2,427 | Word-start token that occurs as a complete standalone word in the **majority** of its occurrences: `standalone_count / freq > 1/2`. |
| `prefix` | 762 | Word-start token mostly used as the first piece of longer words. Includes both pure WordPiece merge intermediates never seen standalone (`adventur`, `butterf`) and genuine-but-minority words whose derived forms dominate (`sleep`: standalone 157 of 345, the rest *sleeping/sleepy*; `t`: mostly the contraction fragment of *didn't*; `ben`: mostly *bench/beneath* despite name_frac 1.0). |
| `suffix` | 798 | `##` continuation piece (word-internal/final: `##ing`, `##ness`, `##s`). |
| `punctuation` | 20 | `. , ! ?` etc. (isolated by pre-tokenization, so always single tokens). |
| `digit` | 10 | `0`–`9`. Numbers are always split into individual digits. |
| `special` | 2 | `[UNK]`, `[EOS]`. |

The `word`/`prefix` split is empirical (sample counts), and the majority rule
means class `prefix` does **not** imply "not a word" — check `standalone_count`
for the full picture; the POS/name columns are computed regardless of class.

## POS tags (spaCy / Universal Dependencies)

NOUN, VERB, ADJ (adjective), ADV (adverb), PROPN (proper noun), PRON (pronoun),
DET (determiner: *the, a*), ADP (adposition/preposition: *in, of*), AUX
(auxiliary: *is, was, can*), PART (particle: *to* [infinitive], *not*, *'s*),
CCONJ / SCONJ (coordinating/subordinating conjunction: *and* / *because*), NUM
(numeral: *two*), INTJ (interjection: *wow*). Everything else (rare tags, tagger
noise) is lumped into `pos_other`.

## Definition of `name_frac`

Only **mid-sentence** occurrences are informative about names (every word is
capitalized sentence-initially), so with $M(w)$ = the set of mid-sentence
standalone occurrences of word $w$:

$$
\texttt{name\_frac}(w) \;=\; \frac{\#\{\,o \in M(w) : \text{capitalized}(o) \,\wedge\, \text{POS}(o)=\text{PROPN}\,\}}{\#M(w)}
$$

"Mid-sentence" = the preceding spaCy token is not sentence-opening punctuation
(`.!?"“”:;(`). Assuming name-vs-common usage is independent of sentence position,
this is an unbiased estimate of the overall name-usage fraction.

Both conditions are needed because each alone has a failure mode:

- capitalization alone counts the pronoun *I* (always capitalized, never a name);
- the PROPN tag alone inherits spaCy-small mistakes — it tags lowercase *awe* as
  PROPN in "They watched in awe." (`awe`: pos_PROPN 0.70 but name_frac 0.00).

Interpretation guide: `name_frac` ≈ 1 → dedicated name token (*lily* 0.97, *mia*,
*leo*, *ben*, …, ~20 tokens); intermediate values → dual-use words (*fluff* 0.83,
*keeper* 0.42, *earth* 0.37, *dad* 0.34, *mom* 0.17); ≈ 0 → common word. A few
rare short tokens score high through genuine but marginal name usage (*t* 0.93
via "T-Rex"; *mu*, *fi* as robot-character names) — filter on `freq` if you only
want the major names.

## Caveats

- All statistics are sample estimates from 20k stories (~3.7 M words); values for
  low-`freq` tokens are noisy. 114 tokens have `freq` = 0 (never occurred in the
  sample at all).
- POS tags come from spaCy's *small* English model (~97% accuracy on standard
  text); systematic errors on unusual constructions will leave traces in the
  distributions, mostly in the tail.
- `[EOS]` shows `freq` 0 because stories were tokenized without the appended
  `[EOS]`; in training it appears once per story. `[UNK]`'s `freq` 0 means the
  sample contains no character outside the tokenizer's alphabet.
- **Contraction/hyphen mismatch** (worst case: token `t`, id 51): the WordPiece
  pre-tokenizer isolates apostrophes, so `didn't` → `didn` `'` `t` — for the
  model, `t` is mostly a contraction fragment (~72% of its `freq`). But spaCy
  segments contractions as `did` + `n't` (skipped as non-alphabetic), so the
  POS/name columns for `t` are instead based on its rare standalone occurrences,
  which are dominated by spaCy splitting `T-Rex` → `T` + `-` + `Rex`: hence
  `name_frac` 0.93 (the majority-standalone rule correctly classes it `prefix`).
  Read single-letter tokens (`t`, `s`, `d`, …) with this in mind: their `freq` is
  mostly contraction/possessive pieces, while their POS/name columns describe
  something rarer.
