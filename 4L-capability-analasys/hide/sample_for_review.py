"""Stratified sample of predictions for qualitative (LLM) review.

Strata (disjoint):
  near_miss          wrong, but true token in top-5           (60)
  wrong_moderate     wrong, rank>5, loss < 8 nats             (140)
  catastrophic       wrong, rank>5, loss >= 8 nats            (60)
  confident_correct  right with p >= 0.8                      (50)
  hesitant_correct   right with p < 0.3                       (30)

Each example carries 60 tokens of decoded context, the true token, the model's
top-5 with probabilities, and whether the (last-token, true-token) bigram
appeared earlier in the context. Examples are dealt round-robin into
review/batches/batch_<i>.json so every batch mixes all strata.
"""

import json

import numpy as np

from common import EOS_ID, HERE, results, vocab_classes, vocab_strings, decode_context

N_BATCHES = 8
STRATA = [  # name, count, mask-builder
    ("near_miss", 60, lambda err, rank, loss, p1: err & (rank <= 5)),
    ("wrong_moderate", 140, lambda err, rank, loss, p1: err & (rank > 5) & (loss < 8)),
    ("catastrophic", 60, lambda err, rank, loss, p1: err & (rank > 5) & (loss >= 8)),
    ("confident_correct", 50, lambda err, rank, loss, p1: ~err & (p1 >= 0.8)),
    ("hesitant_correct", 30, lambda err, rank, loss, p1: ~err & (p1 < 0.3)),
]


def bigram_seen(row: np.ndarray, t: int) -> bool:
    """Did (inputs[t], targets[t]) = (row[t], row[t+1]) occur earlier as a pair?"""
    inp = row[: t + 1]
    return bool(np.any((inp[:-1] == row[t]) & (inp[1:] == row[t + 1])))


def main():
    r = results()
    rows, loss, rank = r["rows"], r["loss"], r["rank"]
    top_ids, top_p = r["top_ids"], r["top_p"].astype(float)
    targets = rows[:, 1:]
    strs = vocab_strings()
    cls = np.array(vocab_classes())

    err = rank > 1
    p1 = top_p[:, :, 0]
    t_grid = np.broadcast_to(np.arange(512), err.shape)
    eligible = (t_grid >= 30) & (targets != EOS_ID) & (cls[targets] != "bytes")

    rng = np.random.default_rng(0)
    examples = []
    for name, count, build in STRATA:
        n_idx, t_idx = np.where(build(err, rank, loss, p1) & eligible)
        pick = rng.choice(len(n_idx), count, replace=False)
        for n, t in zip(n_idx[pick], t_idx[pick]):
            examples.append({
                "id": f"{name}_{n}_{t}",
                "stratum": name,
                "context": decode_context(rows[n], int(t), 60),
                "true_token": strs[targets[n, t]],
                "top5": [{"token": strs[i], "p": round(float(p), 3)}
                         for i, p in zip(top_ids[n, t], top_p[n, t])][:5],
                "loss_nats": round(float(loss[n, t]), 2),
                "true_rank": int(rank[n, t]),
                "bigram_seen_in_context": bigram_seen(rows[n], int(t)),
            })

    rng.shuffle(examples)
    batch_dir = HERE / "review" / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_BATCHES):
        batch = examples[i::N_BATCHES]
        with open(batch_dir / f"batch_{i}.json", "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=1, ensure_ascii=False)
        print(f"batch_{i}.json: {len(batch)} examples")


if __name__ == "__main__":
    main()
