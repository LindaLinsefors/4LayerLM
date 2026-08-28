"""Does a wrong top-1 prediction at least have the right part of speech?

For a sample of error positions: take the local context, append (a) the true
token and (b) the predicted token, POS-tag both variants with spaCy, and
compare the POS of the final word. This measures in-slot syntactic category
agreement -- a stronger check than string-class matching in metrics.py.

Caveat: spaCy tags whatever string it gets, including nonwords from bad
subword continuations, so this is a proxy; the LLM review provides the
complementary judgment. Saves results/pos_check.json.
"""

import json

import numpy as np
import spacy

from common import HERE, results, vocab_strings

N_SAMPLE = 3000
CTX_TOKENS = 30
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}


def main():
    r = results()
    rows, rank, top_ids = r["rows"], r["rank"], r["top_ids"]
    targets = rows[:, 1:]
    pred = top_ids[:, :, 0]
    strs = vocab_strings()

    rng = np.random.default_rng(0)
    err_n, err_t = np.where(rank > 1)
    keep = err_t >= CTX_TOKENS
    err_n, err_t = err_n[keep], err_t[keep]
    pick = rng.choice(len(err_n), N_SAMPLE, replace=False)
    err_n, err_t = err_n[pick], err_t[pick]

    texts, meta = [], []
    for n, t in zip(err_n, err_t):
        ctx_ids = rows[n, t + 1 - CTX_TOKENS : t + 1].tolist()
        ctx = "".join(strs[i] for i in ctx_ids)
        s_true, s_pred = strs[targets[n, t]], strs[pred[n, t]]
        if not s_true.strip() or not s_pred.strip():
            continue  # whitespace tokens have no POS
        texts += [ctx + s_true, ctx + s_pred]
        meta.append((int(n), int(t)))

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    docs = list(nlp.pipe(texts, batch_size=256))

    match = both_content_or_both_function = 0
    confusion = {}
    n_pairs = len(meta)
    for i in range(n_pairs):
        dt, dp = docs[2 * i], docs[2 * i + 1]
        pt, pp = dt[-1].pos_, dp[-1].pos_
        match += pt == pp
        both_content_or_both_function += (pt in CONTENT) == (pp in CONTENT)
        if pt != pp:
            confusion[f"{pt}->{pp}"] = confusion.get(f"{pt}->{pp}", 0) + 1

    out = {
        "n_pairs": n_pairs,
        "pos_match": match / n_pairs,
        "content_function_match": both_content_or_both_function / n_pairs,
        "top_confusions": dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:15]),
    }
    print(f"error positions checked: {n_pairs}")
    print(f"same POS in slot:            {out['pos_match']:.1%}")
    print(f"same content/function class: {out['content_function_match']:.1%}")
    print("top POS confusions (true->predicted):")
    for k, v in out["top_confusions"].items():
        print(f"  {k}: {v}")
    with open(HERE / "results" / "pos_check.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
