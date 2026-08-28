"""Merge the LLM review batches (review/agent_out/*.json) into summary
distributions. Prints tables and saves results/review_summary.json."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from common import HERE

ERR_STRATA = ["near_miss", "wrong_moderate", "catastrophic"]
OK_STRATA = ["confident_correct", "hesitant_correct"]


def pct_table(counter: Counter, total: int) -> dict:
    return {k: f"{v} ({v / total:.0%})" for k, v in counter.most_common()}


def main():
    rows = []
    for f in sorted((HERE / "review" / "agent_out").glob("batch_*.json")):
        rows += json.load(open(f, encoding="utf-8"))
    stratum = lambda r: r["id"].rsplit("_", 2)[0]
    print(f"{len(rows)} judged examples from "
          f"{len(list((HERE / 'review' / 'agent_out').glob('batch_*.json')))} batches")

    out = {}
    # errors: plausibility per stratum
    print("\n=== Plausibility of the model's wrong top-1 guess ===")
    for s in ERR_STRATA + ["ALL_ERRORS"]:
        sub = [r for r in rows if (stratum(r) in ERR_STRATA if s == "ALL_ERRORS"
                                   else stratum(r) == s)]
        if not sub:
            continue
        c = Counter(r.get("plausibility", "?") for r in sub)
        out[f"plausibility/{s}"] = pct_table(c, len(sub))
        print(f"  {s} (n={len(sub)}): ", out[f"plausibility/{s}"])

    print("\n=== Aspects right when wrong (share of errors with each aspect) ===")
    errs = [r for r in rows if stratum(r) in ERR_STRATA]
    c = Counter(a for r in errs for a in r.get("aspects_right", []))
    out["aspects_right"] = {k: f"{v / len(errs):.0%}" for k, v in c.most_common()}
    print("  ", out["aspects_right"])

    print("\n=== What was needed to get the true token (errors) ===")
    for s in ERR_STRATA + ["ALL_ERRORS"]:
        sub = [r for r in rows if (stratum(r) in ERR_STRATA if s == "ALL_ERRORS"
                                   else stratum(r) == s)]
        if not sub:
            continue
        c = Counter(r.get("failure_needs", "?") for r in sub)
        out[f"failure_needs/{s}"] = pct_table(c, len(sub))
        print(f"  {s}: ", out[f"failure_needs/{s}"])

    print("\n=== Why right (correct strata) ===")
    for s in OK_STRATA:
        sub = [r for r in rows if stratum(r) == s]
        c = Counter(r.get("why_right", "?") for r in sub)
        out[f"why_right/{s}"] = pct_table(c, len(sub))
        print(f"  {s} (n={len(sub)}): ", out[f"why_right/{s}"])

    print("\n=== Domain distribution (all examples) ===")
    c = Counter(r.get("domain", "?") for r in rows)
    out["domain"] = pct_table(c, len(rows))
    print("  ", out["domain"])

    print("\n=== Plausibility x domain (errors) ===")
    cross = defaultdict(Counter)
    for r in errs:
        cross[r.get("domain", "?")][r.get("plausibility", "?")] += 1
    for d, c in sorted(cross.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        bad = c["grammatical_only"] + c["broken"]
        print(f"  {d} (n={n}): good-or-plausible {1 - bad / n:.0%}")
    out["plaus_x_domain"] = {d: dict(c) for d, c in cross.items()}

    notes = [(r["id"], r["note"]) for r in rows if r.get("note")]
    out["n_notes"] = len(notes)
    with open(HERE / "results" / "review_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n{len(notes)} examples carry notes; saved results/review_summary.json")


if __name__ == "__main__":
    main()
