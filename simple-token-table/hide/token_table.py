"""Build token_table.csv (+ sorted_token_table.csv, the same table grouped by
class/top_pos/freq): a classification of all 4019 SimpleStories tokenizer tokens.

For every token id: structural class, corpus frequency, whether it is attested as a
standalone word, a POS distribution over its corpus occurrences (via spaCy on raw,
capitalized story text), a proper-name flag, and example words for subword pieces.

POS attribution: each spaCy word occurrence is lowercased and WordPiece-tokenized.
A single-piece word counts its POS tag on that token; a split word counts its POS
on every piece (so `##ing` accumulates the POS of words it terminates).

Name detection (name_frac): fraction of mid-sentence occurrences that are both
capitalized and tagged PROPN. Requiring both filters spaCy's small-model habit of
mis-tagging lowercase nouns as PROPN ("watched in awe.") and capitalized non-names
("I"); sentence-initial occurrences are excluded since capitalization is
uninformative there.

Run: python token_table.py            (~5 min: downloads 17 MB of stories once,
                                       tags ~20k stories with spaCy en_core_web_sm)
"""

import collections
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root, for load.py
from load import load_tokenizer

N_STORIES = 20_000
DATA = Path(__file__).parent / "data"
PARQUET_URL = (
    "https://huggingface.co/datasets/SimpleStories/SimpleStories/resolve/"
    "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
# spaCy POS tags kept as individual columns (rest lumped into pos_other)
POS_COLS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON", "DET", "ADP", "AUX",
            "PART", "CCONJ", "SCONJ", "NUM", "INTJ"]


def get_stories(n: int = N_STORIES) -> list[str]:
    DATA.mkdir(exist_ok=True)
    cache = DATA / "simplestories_test.parquet"
    if not cache.exists():
        print(f"downloading {PARQUET_URL} ...")
        with urllib.request.urlopen(PARQUET_URL, timeout=120) as r:
            cache.write_bytes(r.read())
    stories = pd.read_parquet(cache, columns=["story"])["story"]
    print(f"{len(stories)} stories in shard, using {min(n, len(stories))}")
    return stories.iloc[:n].tolist()


def main() -> None:
    tok = load_tokenizer("simple_2l")
    vocab: dict[str, int] = tok.get_vocab()
    id_to_token = {i: t for t, i in vocab.items()}
    stories = get_stories()

    # --- exact token frequencies: tokenize every story like training did ---
    print("tokenizing corpus ...")
    freq = collections.Counter()
    for enc in tok(stories, add_special_tokens=False)["input_ids"]:
        freq.update(enc)

    # --- POS attribution via spaCy on the raw (capitalized) text ---
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    pos_of = collections.defaultdict(collections.Counter)  # token id -> POS counts
    standalone = collections.Counter()  # id -> occurrences as a complete word
    contains = collections.defaultdict(collections.Counter)  # id -> words using it
    mid = collections.Counter()  # id -> standalone occurrences mid-sentence
    mid_cap = collections.Counter()  # ... of which capitalized
    mid_name = collections.Counter()  # ... of which capitalized AND tagged PROPN
    pieces_cache: dict[str, list[int]] = {}

    print("POS-tagging with spaCy ...")
    for i, doc in enumerate(nlp.pipe(stories, batch_size=256)):
        if i % 2000 == 0:
            print(f"  {i}/{len(stories)}")
        for t in doc:
            if not t.is_alpha:  # skip punctuation and junk like "b4l", "mr."
                continue
            low = t.text.lower()
            ids = pieces_cache.get(low)
            if ids is None:
                ids = pieces_cache[low] = tok.encode(low, add_special_tokens=False)
            if len(ids) == 1:
                standalone[ids[0]] += 1
                if t.i > 0 and doc[t.i - 1].text not in '.!?"“”:;(':
                    mid[ids[0]] += 1
                    if t.text[0].isupper():
                        mid_cap[ids[0]] += 1
                        if t.pos_ == "PROPN":
                            mid_name[ids[0]] += 1
            for tid in ids:
                pos_of[tid][t.pos_] += 1
                if len(ids) > 1:
                    contains[tid][low] += 1

    # --- assemble the table ---
    rows = []
    for tid in range(len(vocab)):
        s = id_to_token[tid]
        if s.startswith("[") and s.endswith("]"):
            cls = "special"
        elif s.startswith("##"):
            cls = "suffix"  # continuation piece
        elif re.fullmatch(r"[a-z]+", s):
            # a word only if it stands alone most of the time it appears;
            # tokens mostly used as the first piece of longer words are prefixes
            # (e.g. `t`: freq dominated by contraction splits didn't -> didn ' t)
            cls = "word" if 2 * standalone[tid] > freq[tid] else "prefix"
        elif s.isdigit():
            cls = "digit"
        else:
            cls = "punctuation"

        pos = pos_of[tid]
        pos_total = sum(pos.values())
        fr = {f"pos_{p}": round(pos[p] / pos_total, 3) if pos_total else None
              for p in POS_COLS}
        fr["pos_other"] = (round(sum(c for p, c in pos.items() if p not in POS_COLS)
                                 / pos_total, 3) if pos_total else None)
        top_pos = max(pos, key=pos.get) if pos else None
        n_mid = mid[tid]
        cap_frac = round(mid_cap[tid] / n_mid, 3) if n_mid >= 3 else None
        name_frac = round(mid_name[tid] / n_mid, 3) if n_mid >= 3 else None
        examples = " ".join(w for w, _ in contains[tid].most_common(3))

        rows.append({
            "id": tid,
            "token": s,
            "class": cls,
            "freq": freq[tid],
            "standalone_count": standalone[tid],
            "top_pos": top_pos,
            "name_frac": name_frac,
            "cap_frac": cap_frac,
            **fr,
            "example_words": examples,
        })

    df = pd.DataFrame(rows)
    here = Path(__file__).parent
    df.to_csv(here.parent / "token_table.csv", index=False)

    # second view: grouped by class (in the order below), then POS, then frequency
    class_order = ["word", "prefix", "suffix", "punctuation", "digit", "special"]
    df_sorted = df.assign(
        **{"class": pd.Categorical(df["class"], categories=class_order, ordered=True)}
    ).sort_values(["class", "top_pos", "freq"], ascending=[True, True, False])
    df_sorted.to_csv(here.parent / "sorted_token_table.csv", index=False)
    print(f"\nwrote token_table.csv and sorted_token_table.csv ({len(df)} rows)")
    print("\nclass counts:")
    print(df["class"].value_counts().to_string())
    print("\ntop_pos counts among class=='word':")
    print(df[df["class"] == "word"]["top_pos"].value_counts().to_string())
    names = df[df["name_frac"] >= 0.5].sort_values("freq", ascending=False)
    print(f"\ntokens with name_frac >= 0.5: {len(names)}")
    print([f"{t} ({f:.2f})" for t, f in zip(names["token"], names["name_frac"])])


if __name__ == "__main__":
    main()
