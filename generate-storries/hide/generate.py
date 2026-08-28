"""Sample 20 random SimpleStories, prompt the 2-layer model with each story's
first 20 tokens, and let it generate a continuation (plain sampling,
temperature 1.0) until it emits [EOS] or fills the 512-token context window.

Outputs (same folder):
  stories.md    -- side-by-side table: shared prompt | original rest | model rest
  results.json  -- raw data: texts, token ids, lengths, stop reason

Run:  python generate-storries/generate.py   (from the project root)
"""

import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from load import _rows, load_simple_2l

OUT_DIR = Path(__file__).resolve().parent
SEED = 0
N_STORIES = 20
PROMPT_LEN = 20  # tokens of each story given to the model
N_CTX = 512  # context window; generation stops when the sequence reaches this
DATASET_RANGE = 100_000  # rows drawn uniformly from this prefix of the train split
TEMPERATURE = 1.0  # sample from the model's own distribution, no top-k

rng = random.Random(SEED)
torch.manual_seed(SEED)

model, _, tok = load_simple_2l()  # tokenizer encodes WITHOUT appending [EOS]
eos_id = tok.convert_tokens_to_ids("[EOS]")
assert tok.convert_ids_to_tokens(eos_id) == "[EOS]"


# --- pick the stories -------------------------------------------------------


def fetch_story(offset: int, tries: int = 3) -> str:
    for attempt in range(tries):
        try:
            return _rows("SimpleStories/SimpleStories", offset, 1)[0]["row"]["story"]
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2)


# a few spare offsets in case a story is too short to have a continuation
offsets = rng.sample(range(DATASET_RANGE), N_STORIES + 10)
stories = []  # (dataset_row, raw_text, token_ids)
for off in offsets:
    if len(stories) == N_STORIES:
        break
    text = fetch_story(off)
    ids = tok.encode(text)
    if len(ids) > PROMPT_LEN:
        stories.append((off, text, ids))
    print(f"row {off}: {len(ids)} tokens")
assert len(stories) == N_STORIES


# --- batched generation (all prompts are exactly PROMPT_LEN tokens) ---------

idx = torch.tensor([ids[:PROMPT_LEN] for _, _, ids in stories])  # (20, 20)
finished = torch.zeros(N_STORIES, dtype=torch.bool)
t0 = time.time()
with torch.no_grad():
    while idx.shape[1] < N_CTX and not finished.all():
        logits = model(idx)[:, -1, :] / TEMPERATURE
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        nxt[finished] = eos_id  # finished rows just pad with [EOS]
        finished |= nxt == eos_id
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        if idx.shape[1] % 64 == 0:
            print(
                f"  {idx.shape[1]}/{N_CTX} tokens, "
                f"{int(finished.sum())}/{N_STORIES} stories ended, {time.time() - t0:.0f}s"
            )
print(f"generation took {time.time() - t0:.0f}s")


# --- decode -----------------------------------------------------------------


def continuation_text(prompt_ids: list[int], cont_ids: list[int]) -> str:
    """Decode a continuation so that a word split mid-token at the prompt
    boundary comes out merged: decode(prompt + continuation) minus the decoded
    prompt prefix. Falls back to a separate decode if the tokenizer's
    punctuation-spacing cleanup breaks the prefix property at the boundary."""
    prompt_text = tok.decode(prompt_ids)
    full = tok.decode(prompt_ids + cont_ids)
    if full.startswith(prompt_text):
        return full[len(prompt_text) :]
    return " " + tok.decode(cont_ids)


results = []
for i, (off, text, ids) in enumerate(stories):
    prompt_ids = ids[:PROMPT_LEN]
    gen = idx[i, PROMPT_LEN:].tolist()
    ended = eos_id in gen
    if ended:
        gen = gen[: gen.index(eos_id)]
    results.append(
        dict(
            story=i + 1,
            dataset_row=off,
            prompt=tok.decode(prompt_ids),
            original_rest=continuation_text(prompt_ids, ids[PROMPT_LEN:]),
            generated_rest=continuation_text(prompt_ids, gen),
            original_raw_text=text,
            original_ids=ids,
            generated_ids=prompt_ids + gen,
            original_len=len(ids),
            generated_len=PROMPT_LEN + len(gen),
            ended_with_eos=ended,
        )
    )

(OUT_DIR / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


# --- write the markdown report ----------------------------------------------


def md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", "<br>")


n_eos = sum(r["ended_with_eos"] for r in results)
mean_orig = sum(r["original_len"] for r in results) / N_STORIES
mean_gen = sum(r["generated_len"] for r in results) / N_STORIES

lines = [
    "# SimpleStories 2-layer model: story continuations",
    "",
    f"20 stories drawn at random (seed {SEED}) from the first {DATASET_RANGE:,} rows of the",
    "`SimpleStories/SimpleStories` train split. The model was given each story's first",
    f"{PROMPT_LEN} tokens and generated until it emitted `[EOS]` or the sequence reached the",
    f"{N_CTX}-token context window. Sampling: temperature {TEMPERATURE}, no top-k",
    "(i.e. the model's own next-token distribution), torch seed 0.",
    "",
    f"{n_eos}/{N_STORIES} generations ended with `[EOS]`. Mean length in tokens: original",
    f"{mean_orig:.0f}, generated {mean_gen:.0f} (both include the {PROMPT_LEN}-token prompt).",
    "",
    "Both text columns are *decoded tokens* — the tokenizer lowercases and normalizes",
    "spacing, so the original reads slightly differently from the raw dataset text",
    "(which is preserved in `results.json`).",
    "",
    "| # | dataset row | original tokens | generated tokens | generation stopped by |",
    "|--:|--:|--:|--:|:--|",
]
for r in results:
    stop = "`[EOS]`" if r["ended_with_eos"] else "context window"
    lines.append(
        f"| {r['story']} | {r['dataset_row']} | {r['original_len']} "
        f"| {r['generated_len']} | {stop} |"
    )

lines += [
    "",
    "## Side by side",
    "",
    "Each row: the shared 20-token prompt, then the original story's continuation next",
    "to the model's sampled continuation.",
    "",
    "| # | prompt (first 20 tokens) | original continuation | model continuation |",
    "|--:|:--|:--|:--|",
]
for r in results:
    gen = md(r["generated_rest"])
    if not r["ended_with_eos"]:
        gen += " *…(hit context limit)*"
    lines.append(f"| {r['story']} | {md(r['prompt'])} | {md(r['original_rest'])} | {gen} |")

(OUT_DIR.parent / "stories.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT_DIR / 'stories.md'} and results.json")
