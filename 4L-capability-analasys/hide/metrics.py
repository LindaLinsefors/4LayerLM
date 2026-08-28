"""Automatic (non-LLM) metrics over the recorded predictions.

Sections:
  1. Overall accuracy / loss
  2. Accuracy by class of the true token (word start/continuation, punct, ...)
  3. Copying & induction: accuracy conditioned on whether the answer was
     completable from the context (bigram seen / token seen / novel)
  4. Calibration: top-1 confidence vs empirical accuracy
  5. Error anatomy: partial credit on wrong top-1 predictions (spacing, case,
     token class, near-duplicates, near-misses in top-5/10), echo/anticipation
  6. Frequency & entropy effects, doc-distance breakdown

Prints a readable report and saves results/metrics.json.
"""

import json

import numpy as np

from common import EOS_ID, results, vocab_classes, vocab_strings, HERE

J = {}  # everything we print also lands here -> metrics.json


def sec(title):
    print(f"\n=== {title} ===")


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header, None] + rows:
        if r is None:
            print("  " + "-+-".join("-" * w for w in widths))
        else:
            print("  " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def main():
    r = results()
    rows, loss, rank = r["rows"], r["loss"], r["rank"]
    entropy, top_ids, top_p = r["entropy"], r["top_ids"], r["top_p"].astype(np.float32)
    inputs, targets = rows[:, :-1], rows[:, 1:]
    pred = top_ids[:, :, 0]
    p1 = top_p[:, :, 0]
    correct = rank == 1
    N = correct.size
    strs = vocab_strings()
    cls = np.array(vocab_classes())

    # ---- 1. overall ----
    sec("Overall")
    J["overall"] = {
        "n_predictions": int(N),
        "top1_acc": float(correct.mean()),
        "top5_acc": float((rank <= 5).mean()),
        "top10_acc": float((rank <= 10).mean()),
        "mean_loss_nats": float(loss.mean()),
        "median_loss_nats": float(np.median(loss)),
        "perplexity": float(np.exp(loss.mean())),
        "median_rank": int(np.median(rank)),
    }
    for k, v in J["overall"].items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- 2. by class of the true token ----
    sec("By class of the true token")
    ends_alnum = np.array([bool(s) and s[-1].isalnum() for s in strs])
    tcls = cls[targets].copy()
    nospace = tcls == "alpha_nospace"
    tcls[nospace & ends_alnum[inputs]] = "word_cont"
    tcls[nospace & ~ends_alnum[inputs]] = "word_start_nospace"
    tcls[targets == EOS_ID] = "eos"
    out = []
    for c in ["alpha_space", "word_cont", "word_start_nospace", "punct", "number",
              "space", "mixed", "bytes", "eos"]:
        m = tcls == c
        if m.sum() == 0:
            continue
        out.append([c, int(m.sum()), f"{m.mean():.1%}", f"{correct[m].mean():.1%}",
                    f"{(rank[m] <= 10).mean():.1%}", f"{loss[m].mean():.2f}"])
    table(out, ["class", "count", "share", "top1", "top10", "loss"])
    J["by_class"] = {o[0]: {"count": o[1], "top1": o[3], "top10": o[4], "loss": o[5]}
                     for o in out}

    # ---- 3. copying & induction ----
    sec("Copying & induction (was the answer visible in the context?)")
    cond = np.zeros(targets.shape, dtype=np.int8)  # 0 novel, 1 seen-token, 2 bigram
    for n in range(rows.shape[0]):
        seen, follow = set(), {}
        inp, tgt = inputs[n], targets[n]
        for t in range(inp.shape[0]):
            if t >= 1:
                follow.setdefault(inp[t - 1], set()).add(inp[t])
            seen.add(inp[t])
            if tgt[t] in follow.get(inp[t], ()):
                cond[n, t] = 2
            elif tgt[t] in seen:
                cond[n, t] = 1
    out = []
    for v, name in [(2, "bigram seen in context"), (1, "token seen, bigram not"),
                    (0, "token novel in context")]:
        m = cond == v
        out.append([name, int(m.sum()), f"{m.mean():.1%}", f"{correct[m].mean():.1%}",
                    f"{loss[m].mean():.2f}"])
    table(out, ["condition", "count", "share", "top1", "loss"])
    print(f"  share of all CORRECT predictions with bigram seen: "
          f"{(cond[correct] == 2).mean():.1%}")
    print(f"  share of all predictions:                          {(cond == 2).mean():.1%}")
    J["copying"] = {o[0]: {"count": o[1], "share": o[2], "top1": o[3], "loss": o[4]}
                    for o in out}
    J["copying"]["correct_with_bigram_seen"] = float((cond[correct] == 2).mean())

    # ---- 4. calibration ----
    sec("Calibration (top-1 probability vs accuracy of the top-1 guess)")
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p1 >= lo) & (p1 < hi)
        if m.sum():
            out.append([f"[{lo:.1f},{hi:.1f})", int(m.sum()),
                        f"{p1[m].mean():.2f}", f"{correct[m].mean():.1%}"])
    table(out, ["conf bin", "count", "mean conf", "top1 acc"])
    ece = sum(int(o[1]) * abs(float(o[2]) - float(o[3].rstrip('%')) / 100) for o in out) / N
    print(f"  expected calibration error: {ece:.3f}")
    J["calibration"] = {"bins": out, "ece": float(ece)}

    # ---- 5. error anatomy ----
    sec("Error anatomy (top-1 wrong)")
    err = ~correct
    ts = np.array(strs, dtype=object)
    s_true, s_pred = ts[targets[err]], ts[pred[err]]
    lead = lambda a: np.array([s.startswith(" ") for s in a])

    def first_alpha_case(a):
        out = np.full(len(a), -1, dtype=np.int8)  # -1 no alpha, 0 lower, 1 upper
        for i, s in enumerate(a):
            for ch in s:
                if ch.isalpha():
                    out[i] = ch.isupper()
                    break
        return out

    core = lambda a: np.array([s.strip().lower() for s in a], dtype=object)
    ct, cp = core(s_true), core(s_pred)
    case_t, case_p = first_alpha_case(s_true), first_alpha_case(s_pred)
    cls_t, cls_p = cls[targets[err]], cls[pred[err]]
    both_alpha = (case_t >= 0) & (case_p >= 0)
    near_dup = ct == cp
    prefix = np.array([len(a) >= 2 and len(b) >= 2 and a != b
                       and (a.startswith(b) or b.startswith(a))
                       for a, b in zip(ct, cp)])
    stats = {
        "n_errors": int(err.sum()),
        "true_in_top5": float((rank[err] <= 5).mean()),
        "true_in_top10": float((rank[err] <= 10).mean()),
        "leading_space_match": float((lead(s_true) == lead(s_pred)).mean()),
        "class_match": float((cls_t == cls_p).mean()),
        "case_match_given_both_alpha": float((case_t == case_p)[both_alpha].mean()),
        "both_alpha_share": float(both_alpha.mean()),
        "near_duplicate (same word, case/space differ)": float(near_dup.mean()),
        "prefix_relation (one extends the other)": float(prefix.mean()),
        "echo (pred = current token)": float((pred[err] == inputs[err]).mean()),
    }
    ant = pred[:, :-1][err[:, :-1]] == targets[:, 1:][err[:, :-1]]
    stats["anticipation (pred = the token AFTER the true one)"] = float(ant.mean())
    for k, v in stats.items():
        print(f"  {k}: {v:.1%}" if isinstance(v, float) else f"  {k}: {v}")
    J["errors"] = stats

    # ---- 6. frequency, entropy, doc distance ----
    sec("Accuracy by corpus frequency of the true token (counts in these 400 rows)")
    freq = np.bincount(targets.ravel(), minlength=len(strs))
    f = freq[targets]
    out = []
    for lo, hi, name in [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 100, "21-100"),
                         (101, 1000, "101-1k"), (1001, 10**9, ">1k")]:
        m = (f >= lo) & (f <= hi)
        out.append([name, int(m.sum()), f"{correct[m].mean():.1%}", f"{loss[m].mean():.2f}"])
    table(out, ["freq", "count", "top1", "loss"])
    J["by_frequency"] = {o[0]: {"count": o[1], "top1": o[2], "loss": o[3]} for o in out}

    sec("Predictive entropy")
    print(f"  mean entropy when correct: {entropy[correct].mean():.2f} nats")
    print(f"  mean entropy when wrong:   {entropy[err].mean():.2f} nats")
    print(f"  corr(entropy, loss): {np.corrcoef(entropy.ravel(), loss.ravel())[0,1]:.3f}")
    J["entropy"] = {"mean_correct": float(entropy[correct].mean()),
                    "mean_wrong": float(entropy[err].mean()),
                    "corr_with_loss": float(np.corrcoef(entropy.ravel(), loss.ravel())[0, 1])}

    sec("By within-document context length (tokens since last EOS; -1 = doc started before row)")
    idx = np.arange(inputs.shape[1])
    last = np.maximum.accumulate(np.where(inputs == EOS_ID, idx, -1), axis=1)
    dist = np.where(last >= 0, idx - last, -1)
    out = []
    for lo, hi, name in [(-1, -1, "-1 (truncated)"), (0, 0, "0 (cold open)"),
                         (1, 10, "1-10"), (11, 50, "11-50"), (51, 200, "51-200"),
                         (201, 10**9, ">200")]:
        m = (dist >= lo) & (dist <= hi)
        out.append([name, int(m.sum()), f"{correct[m].mean():.1%}", f"{loss[m].mean():.2f}"])
    table(out, ["doc dist", "count", "top1", "loss"])
    J["by_doc_distance"] = {o[0]: {"count": o[1], "top1": o[2], "loss": o[3]} for o in out}

    with open(HERE / "results" / "metrics.json", "w") as fjson:
        json.dump(J, fjson, indent=1)
    print(f"\nsaved results/metrics.json")


if __name__ == "__main__":
    main()
