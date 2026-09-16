# L0_k — component groups by expected CI co-firing (h.0.attn.k_proj)

All 126 alive components (harvest mean CI > 1e-6). Groups were formed by Claude (2026-08-28) reading each component's **full website description** (label + autointerp reasoning, fetched from the paper site) and grouping components expected to have high per-token CI-similarity: same trigger tokens/positions, subset-detectors of one pattern merged; patterns on different tokens kept apart; bimodal or vague descriptions left ungrouped. No activation/CI data was used to form the groups.

Per group with >1 member, four pairwise grids over a 2,048,000-token Pile sample (4000 cached rows):

- **co-activation** — Pearson r of |a_c| = |‖U_c‖ (V_c·φ)| per token
- **co-CI** — Pearson r of the causal importance (lower_leaky) per token; gray = component never varied (no CI in sample)
- **input cosine** — |cos(V_a, V_b)| of read-in vectors (|·|: component sign is gauge)
- **output cosine** — |cos(U_a, U_b)| of write vectors (|·|: same gauge)

'fires' below = tokens with CI > 0.1 in the sample. Within each group, components are ordered by component id (in the lists and on all grid axes).

## Groups

| group | n | members |
|---|---|---|
| [Fires on <|endoftext|>](#eos) | 12 | 18 139 184 193 203 235 305 306 378 421 457 503 |
| ['q'/'Q' predicting ':'](#q-colon) | 13 | 28 64 77 149 234 291 301 311 337 374 471 490 508 |
| ['A' predicting ':'](#a-colon) | 3 | 205 389 443 |
| ['which' predicting ' is'](#which-is) | 4 | 26 128 315 366 |
| ['see' predicting 'also'](#see-also) | 4 | 263 275 409 475 |
| [Wikipedia footer section headers](#wiki-headers) | 2 | 65 210 |
| [Fires on ' of'](#of) | 3 | 9 141 266 |
| [Fires on prepositions (in/at/on/with/by/from)](#prepositions) | 2 | 91 349 |
| [Articles & determiners](#articles) | 3 | 273 353 498 |
| [Loop/control-flow keywords predicting '('](#for-loop) | 5 | 53 55 85 302 442 |
| [Code declaration keywords (public/func/def …)](#decl-keywords) | 2 | 79 215 |
| [LaTeX commands predicting braces](#latex-cmd) | 4 | 187 298 406 461 |
| [Numbers predicting punctuation/units](#num-punct) | 3 | 14 31 413 |
| [URL scheme '://' → domains](#url) | 3 | 161 196 460 |
| [Numbers/abbreviations/initials predicting periods](#abbrev-period) | 2 | 238 403 |
| [Apostrophes predicting contraction endings](#apostrophe) | 2 | 72 286 |
| [Opening quotes & formatting marks](#open-quote) | 2 | 308 332 |
| ['(without) replacement' predicting ' from'](#replacement-from) | 2 | 192 294 |
| ['not'/'no' predicting continuations](#not-cont) | 3 | 45 381 433 |
| [Non-English European tokens](#non-english) | 3 | 127 243 452 |
| [Trailing hyphen/space at line end predicting newline](#eol-newline) | 2 | 319 365 |
| [Horizontal-rule separators predicting newline](#hrule-newline) | 2 | 92 336 |
| [Words that precede ' as'](#before-as) | 2 | 295 504 |
| [Broadly active on most tokens](#dense) | 2 | 29 309 |
| [Ungrouped](#ungrouped) | 41 | 7 13 20 37 67 73 76 90 97 100 109 110 116 125 126 152 191 200 201 232 257 274 287 288 299 331 335 369 371 375 377 404 410 414 415 419 431 445 463 465 494 |

<a id="eos"></a>

### Fires on <|endoftext|> (12)

- **18** (mean CI 0.0001, fires 477): fires on <|endoftext|> to predict document starts
- **139** (mean CI 0.0007, fires 1680): activates on document boundaries and section starters
- **184** (mean CI 0.0006, fires 1457): fires purely on <|endoftext|>
- **193** (mean CI 0.0002, fires 637): fires on the <|endoftext|> token
- **203** (mean CI 0.0006, fires 1420): document separator (<|endoftext|>) detector
- **235** (mean CI 0.0004, fires 1138): fires on the <|endoftext|> token
- **305** (mean CI 0.0002, fires 592): fires on the end of text token
- **306** (mean CI 0.0003, fires 981): fires on <|endoftext|>, predicts new document starts
- **378** (mean CI 0.0004, fires 1404): fires on the <|endoftext|> separator token
- **421** (mean CI 0.0004, fires 1371): fires on the <|endoftext|> document separator token
- **457** (mean CI 0.0002, fires 637): fires exclusively on the <|endoftext|> separator token
- **503** (mean CI 0.0010, fires 2481): fires on the <|endoftext|> token

![Fires on <|endoftext|>](../hide/figures/L0-Attn-k/eos.png)

<a id="q-colon"></a>

### 'q'/'Q' predicting ':' (13)

- **28** (mean CI 0.0001, fires 264): predicts ':' after 'q' to format 'q:'
- **64** (mean CI 0.0001, fires 243): fires on 'q' to predict ':' in q&a context
- **77** (mean CI 0.0001, fires 244): the token 'q' predicting ':' in q&a formats
- **149** (mean CI 0.0001, fires 245): the 'q' token predicting ':' in q&a format
- **234** (mean CI 0.0001, fires 243): the 'Q' token predicting ':' in Q&A formats
- **291** (mean CI 0.0001, fires 242): predicts ':' after 'q' at start of document
- **301** (mean CI 0.0001, fires 252): predicts ':' after 'q' and '>' after 'ubottu'
- **311** (mean CI 0.0001, fires 243): predicting ':' after 'Q' at document start
- **337** (mean CI 0.0001, fires 243): fires on 'q' at start of document to predict ':'
- **374** (mean CI 0.0001, fires 243): the token 'q' predicting ':'
- **471** (mean CI 0.0001, fires 242): predicts ':' after 'q' in q&a format contexts
- **490** (mean CI 0.0001, fires 269): predicts ':' after 'Q' or 'See'
- **508** (mean CI 0.0001, fires 243): token 'q' predicting ':' in q&a format

!['q'/'Q' predicting ':'](../hide/figures/L0-Attn-k/q-colon.png)

<a id="a-colon"></a>

### 'A' predicting ':' (3)

- **205** (mean CI 0.0001, fires 307): the token 'a' predicting a colon (mainly q&a)
- **389** (mean CI 0.0003, fires 956): predicts colon after 'a' in q&a formats
- **443** (mean CI 0.0001, fires 329): the 'a' token preceding a colon in q&a formats

!['A' predicting ':'](../hide/figures/L0-Attn-k/a-colon.png)

<a id="which-is"></a>

### 'which' predicting ' is' (4)

- **26** (mean CI 0.0000, fires 103): the token 'Which' predicting ' is' in math problems
- **128** (mean CI 0.0000, fires 103): predicts " is" after "which"
- **315** (mean CI 0.0000, fires 103): fires on 'which' to predict ' is'
- **366** (mean CI 0.0000, fires 105): predicts ' is' after 'which' and '{' after 'xymatrix'

!['which' predicting ' is'](../hide/figures/L0-Attn-k/which-is.png)

<a id="see-also"></a>

### 'see' predicting 'also' (4)

- **263** (mean CI 0.0000, fires 24): fires on 'see' in 'see also' section headers
- **275** (mean CI 0.0000, fires 32): Predicts 'also' after 'See'
- **409** (mean CI 0.0000, fires 32): predicts 'also' after 'see' in referential contexts
- **475** (mean CI 0.0000, fires 30): predicts ' also' after 'see' in 'see also' headings

!['see' predicting 'also'](../hide/figures/L0-Attn-k/see-also.png)

<a id="wiki-headers"></a>

### Wikipedia footer section headers (2)

- **65** (mean CI 0.0001, fires 207): wikipedia footer section headers
- **210** (mean CI 0.0000, fires 79): predicts second word of wikipedia section headers

![Wikipedia footer section headers](../hide/figures/L0-Attn-k/wiki-headers.png)

<a id="of"></a>

### Fires on ' of' (3)

- **9** (mean CI 0.0000, fires 64): activates on ' of' and predicts following determiners
- **141** (mean CI 0.0003, fires 1053): fires on variations of the word "of"
- **266** (mean CI 0.0000, fires 195): fires on the token ' of'

![Fires on ' of'](../hide/figures/L0-Attn-k/of.png)

<a id="prepositions"></a>

### Fires on prepositions (in/at/on/with/by/from) (2)

- **91** (mean CI 0.0036, fires 11020): fires on prepositions
- **349** (mean CI 0.0023, fires 6151): capitalized prepositions beginning introductory phrases

![Fires on prepositions (in/at/on/with/by/from)](../hide/figures/L0-Attn-k/prepositions.png)

<a id="articles"></a>

### Articles & determiners (3)

- **273** (mean CI 0.0000, fires 258): determiners (articles, possessives, quantifiers)
- **353** (mean CI 0.0004, fires 1281): fires on articles ('the', 'a', 'an')
- **498** (mean CI 0.0003, fires 1013): fires on articles 'the' and 'a'

![Articles & determiners](../hide/figures/L0-Attn-k/articles.png)

<a id="for-loop"></a>

### Loop/control-flow keywords predicting '(' (5)

- **53** (mean CI 0.0001, fires 182): predicts open parenthesis or space after 'for' loops
- **55** (mean CI 0.0001, fires 185): predicts loop initialization syntax after 'for' keywords
- **85** (mean CI 0.0005, fires 1285): predicts opening parenthesis after control flow keywords
- **302** (mean CI 0.0001, fires 189): for/foreach loop syntax initiation
- **442** (mean CI 0.0001, fires 175): fires on 'for' or 'foreach' loop keywords in code

![Loop/control-flow keywords predicting '('](../hide/figures/L0-Attn-k/for-loop.png)

<a id="decl-keywords"></a>

### Code declaration keywords (public/func/def …) (2)

- **79** (mean CI 0.0001, fires 455): fires on code access modifiers (public/private/protected)
- **215** (mean CI 0.0001, fires 230): fires on function and class declaration keywords

![Code declaration keywords (public/func/def …)](../hide/figures/L0-Attn-k/decl-keywords.png)

<a id="latex-cmd"></a>

### LaTeX commands predicting braces (4)

- **187** (mean CI 0.0003, fires 1011): latex math commands predicting braces and subscripts
- **298** (mean CI 0.0003, fires 827): latex fraction commands (\frac) predicting opening brace
- **406** (mean CI 0.0005, fires 1124): predicts '{' after latex commands like begin and end
- **461** (mean CI 0.0035, fires 9573): latex formatting macros and commands in math mode

![LaTeX commands predicting braces](../hide/figures/L0-Attn-k/latex-cmd.png)

<a id="num-punct"></a>

### Numbers predicting punctuation/units (3)

- **14** (mean CI 0.0001, fires 278): attention key component firing on numbers/digits
- **31** (mean CI 0.0001, fires 520): fires on numeric tokens predicting punctuation or units
- **413** (mean CI 0.0001, fires 415): fires on numbers to predict punctuation or units

![Numbers predicting punctuation/units](../hide/figures/L0-Attn-k/num-punct.png)

<a id="url"></a>

### URL scheme '://' → domains (3)

- **161** (mean CI 0.0001, fires 491): fires on '://' in URLs
- **196** (mean CI 0.0000, fires 176): predicts typical domain prefixes after '://' in urls
- **460** (mean CI 0.0002, fires 648): predicts domain names after url scheme (://)

![URL scheme '://' → domains](../hide/figures/L0-Attn-k/url.png)

<a id="abbrev-period"></a>

### Numbers/abbreviations/initials predicting periods (2)

- **238** (mean CI 0.0099, fires 24853): predicting periods after numbers, abbreviations, and urls
- **403** (mean CI 0.0079, fires 20534): fires on initials, abbreviations, and numbers predicting periods

![Numbers/abbreviations/initials predicting periods](../hide/figures/L0-Attn-k/abbrev-period.png)

<a id="apostrophe"></a>

### Apostrophes predicting contraction endings (2)

- **72** (mean CI 0.0002, fires 639): apostrophes predicting contraction endings
- **286** (mean CI 0.0017, fires 4298): apostrophes predicting contraction and possessive endings

![Apostrophes predicting contraction endings](../hide/figures/L0-Attn-k/apostrophe.png)

<a id="open-quote"></a>

### Opening quotes & formatting marks (2)

- **308** (mean CI 0.0017, fires 4749): opening formatting marks and quotes
- **332** (mean CI 0.0023, fires 5972): opening quotation marks and string delimiters

![Opening quotes & formatting marks](../hide/figures/L0-Attn-k/open-quote.png)

<a id="replacement-from"></a>

### '(without) replacement' predicting ' from' (2)

- **192** (mean CI 0.0000, fires 53): predicts ' from' after ' replacement' in probability problems
- **294** (mean CI 0.0000, fires 53): predicts ' from' after 'without replacement'

!['(without) replacement' predicting ' from'](../hide/figures/L0-Attn-k/replacement-from.png)

<a id="not-cont"></a>

### 'not'/'no' predicting continuations (3)

- **45** (mean CI 0.0003, fires 1132): fires on negative words to predict continuations
- **381** (mean CI 0.0000, fires 207): predicts common continuations of 'no'
- **433** (mean CI 0.0000, fires 64): predicts 'only' or 'just' after 'not'

!['not'/'no' predicting continuations](../hide/figures/L0-Attn-k/not-cont.png)

<a id="non-english"></a>

### Non-English European tokens (3)

- **127** (mean CI 0.0013, fires 3615): non-english european language tokens
- **243** (mean CI 0.0001, fires 285): non-english european function words
- **452** (mean CI 0.0028, fires 7780): non-english european languages and accented characters

![Non-English European tokens](../hide/figures/L0-Attn-k/non-english.png)

<a id="eol-newline"></a>

### Trailing hyphen/space at line end predicting newline (2)

- **319** (mean CI 0.0000, fires 82): predicts newline after a hyphen
- **365** (mean CI 0.0004, fires 1325): predicts newlines after trailing spaces, hyphens, or backslashes

![Trailing hyphen/space at line end predicting newline](../hide/figures/L0-Attn-k/eol-newline.png)

<a id="hrule-newline"></a>

### Horizontal-rule separators predicting newline (2)

- **92** (mean CI 0.0001, fires 132): fires on comment separator lines predicting newline
- **336** (mean CI 0.0010, fires 2795): markdown and code structural boundaries

![Horizontal-rule separators predicting newline](../hide/figures/L0-Attn-k/hrule-newline.png)

<a id="before-as"></a>

### Words that precede ' as' (2)

- **295** (mean CI 0.0011, fires 3070): predicts common fixed continuations like 'as' and 'to'
- **504** (mean CI 0.0006, fires 1830): fires on words that precede ' as'

![Words that precede ' as'](../hide/figures/L0-Attn-k/before-as.png)

<a id="dense"></a>

### Broadly active on most tokens (2)

- **29** (mean CI 0.4686, fires 1179092): broadly active on most tokens
- **309** (mean CI 0.3723, fires 865108): fires on common stop words and punctuation

![Broadly active on most tokens](../hide/figures/L0-Attn-k/dense.png)

<a id="ungrouped"></a>

### Ungrouped (41)

- **7** (mean CI 0.0005, fires 1562): citation identifier prefix predictor
- **13** (mean CI 0.0003, fires 600): fires on '~' indicating subscripts in scientific text
- **20** (mean CI 0.0003, fires 973): predicts '{' after 'frac' and '.' after 'doi'/'pone'
- **37** (mean CI 0.0001, fires 169): fires on 'func' and 'which'
- **67** (mean CI 0.0001, fires 327): q and copyright token predictor
- **73** (mean CI 0.0001, fires 187): fires on markdown image syntax to predict captions
- **76** (mean CI 0.0070, fires 19864): indentation spaces and tabs in code
- **90** (mean CI 0.0006, fires 1753): foreign languages and the word 'not'
- **97** (mean CI 0.0034, fires 10943): comma in a list of items or arguments
- **100** (mean CI 0.0000, fires 47): fires on 'xe' in hex escapes, predicts '2'
- **109** (mean CI 0.0002, fires 377): predicts ':' after 'q' and 'also' after 'see'
- **110** (mean CI 0.0030, fires 7673): html/xml tag and structural boundary marker
- **116** (mean CI 0.0001, fires 220): predicts continuations for 'each', 'package', and 'public'
- **125** (mean CI 0.0005, fires 961): predicts '{' or newline after document structural markers
- **126** (mean CI 0.0006, fires 1519): document boundaries and 'Q' in Q&A starts
- **152** (mean CI 0.0001, fires 214): markdown image tags and 'see also' references
- **191** (mean CI 0.0002, fires 403): predicts syntax (especially colon) following structural keywords
- **200** (mean CI 0.0002, fires 545): fires on superscript carets to predict superscript contents
- **201** (mean CI 0.0004, fires 998): predicts 'd' after series numbers in legal citations
- **232** (mean CI 0.0003, fires 578): structural markers and section headers
- **257** (mean CI 0.0001, fires 304): opening parentheses and braces
- **274** (mean CI 0.0007, fires 2370): predicting the second word in common collocations
- **287** (mean CI 0.0009, fires 2743): structural boundary and document separator tokens
- **288** (mean CI 0.0004, fires 927): predicts a colon after 'category' and 'q'
- **299** (mean CI 0.0740, fires 192780): punctuation symbols and highly specialized technical tokens
- **331** (mean CI 0.0008, fires 3583): all-caps words
- **335** (mean CI 0.0006, fires 1459): urls and categories (negative) vs latex integrals (positive)
- **369** (mean CI 0.0001, fires 249): bimodal: 'func' promoting '(' and 'not' promoting 'only'
- **371** (mean CI 0.0054, fires 14437): document boundaries and task beginnings
- **375** (mean CI 0.0002, fires 772): fires on conjunctions "and" and "or"
- **377** (mean CI 0.0014, fires 3916): fires on 'as', 'so', and 'how'
- **404** (mean CI 0.0014, fires 3940): fires on non-english words and math probabilities
- **410** (mean CI 0.0000, fires 57): fires on 'end' and 'wind' to predict 'up'
- **414** (mean CI 0.0003, fires 986): predicts 'than' after comparatives
- **415** (mean CI 0.0000, fires 68): predicts ' from' after ' replacement' and ':' after 'abstract'
- **419** (mean CI 0.0099, fires 28469): fires on period / dot tokens
- **431** (mean CI 0.0007, fires 1648): fires on '<' in chat logs and ' into'
- **445** (mean CI 0.0001, fires 269): predicts ':' after 'q' and 'also' after 'see'
- **463** (mean CI 0.0001, fires 432): bimodal: html attributes predicting '="' vs which/world/pronouns
- **465** (mean CI 0.0376, fires 85364): newline tokens predicting formatting and indentation
- **494** (mean CI 0.0008, fires 1865): predicts standard trailing punctuation or symbols for specific keywords

## All pairs

All alive components in group order (black lines: group boundaries; last block: ungrouped).

![all pairs](../hide/figures/L0-Attn-k/all_pairs.png)

## Full co-CI grid

The co-CI panel alone, large, with every component index on both axes (same group order as above).

![full co-CI grid](../hide/figures/L0-Attn-k/co_ci_full.png)
