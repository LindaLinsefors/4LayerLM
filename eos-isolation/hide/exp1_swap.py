"""Experiment 1 — context-swap sensitivity: how much do post-EOS predictions
causally depend on what stood before the <|endoftext|>?

For every qualifying mid-sequence EOS event (boundary at p, context X = tokens
0..p-1, fresh document D = tokens p+1..p+64) build two matched input pairs

    EOS pair:  [X, EOS, D]   vs  [X', EOS, D]
    \n  pair:  [X, '\n', D]  vs  [X', '\n', D]

X' is a different, equally long context from another row, ending at a genuine
document end (the p tokens before that row's own EOS), so [X', EOS] stays
in-distribution. The '\n' pair applies the *same* intervention with the
separator's firewall (if any) removed. A third condition anchors the scale:
mid-document swaps [X, S] vs [X', S] on EOS-free rows, where the suffix S
genuinely continues X (maximal true context dependence).

At each offset d = 1..64 after the boundary, with P = P(. | original) and
Q = P(. | swapped context) at the same position:

    KL_d   = sum_v P log(P/Q)          (nats)
    dNLL_d = -log Q(x_d) + log P(x_d)  (true-token log-prob change)
    flip_d = [argmax P != argmax Q]

Perfect disregarding of pre-EOS context <=> KL = 0 identically for the EOS pair.

Cache: hide/cache/exp1.npz (delete to recompute). Figure: ../exp1_swap.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (C_CEIL, C_EOS, C_NL, D, EOS_ID, HERE, NL_ID, ROOT, T,
                    assign_partners, find_events, load_rows, style)

import sys
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 8
N_CEIL = 400
CACHE = HERE / "cache" / "exp1.npz"


def compute():
    rows = load_rows()
    events = find_events(rows)
    rng = np.random.default_rng(0)
    partners = assign_partners(events, rng)
    keep = [i for i in range(len(events)) if partners[i] >= 0]
    print(f"{len(events)} events, {len(keep)} with a replacement-context partner")

    # ceiling: EOS-free rows with pseudo-boundaries drawn from the event-p distribution
    eos_free = np.nonzero(~(rows == EOS_ID).any(1).numpy())[0]
    ceil_rows = rng.choice(eos_free, size=N_CEIL, replace=False)
    ceil_ps = rng.choice(np.array([p for _, p in events]), size=N_CEIL)
    ceil_partners = assign_partners(events, rng, need_p=ceil_ps, own_rows=ceil_rows)
    assert (ceil_partners >= 0).all()

    def repl_ctx(j, length):
        rj, pj = events[j]
        return rows[rj, pj - length:pj]

    units, seqs, seq_start = [], [], []

    def add(unit, member_seqs):
        units.append(unit)
        for s in member_seqs:
            assert len(s) == T
            seqs.append(s)
            seq_start.append(unit["start"])

    for i in keep:
        r, p = events[i]
        row = rows[r]
        rep = repl_ctx(partners[i], p)
        orig_nl = row.clone(); orig_nl[p] = NL_ID
        swap_e = torch.cat([rep, torch.tensor([EOS_ID]), row[p + 1:]])
        swap_nl = torch.cat([rep, torch.tensor([NL_ID]), row[p + 1:]])
        add({"kind": "event", "n": 4, "start": p, "true": row[p + 1:p + 1 + D]},
            [row, swap_e, orig_nl, swap_nl])
    for r, p, j in zip(ceil_rows, ceil_ps, ceil_partners):
        row = rows[r]
        p = int(p)
        swap = torch.cat([repl_ctx(j, p), row[p:]])
        add({"kind": "ceil", "n": 2, "start": p - 1, "true": row[p:p + D]},
            [row, swap])

    model, _, _ = load_pile_4l()
    model = model.to(DEVICE)

    res = {k: [] for k in ["kl_eos", "kl_eos_rev", "kl_nl", "kl_nl_rev",
                           "dnll_eos", "dnll_nl", "flip_eos", "flip_nl",
                           "kl_ceil", "kl_ceil_rev", "dnll_ceil", "flip_ceil"]}
    d_idx = torch.arange(D, device=DEVICE)

    def process(u, lp):
        true = u["true"].to(DEVICE)
        nll = -lp[:, d_idx, true]
        top1 = lp.argmax(-1)

        def kl(a, b):
            return ((lp[a].exp() * (lp[a] - lp[b])).sum(-1)).cpu().numpy()

        if u["kind"] == "event":
            res["kl_eos"].append(kl(0, 1)); res["kl_eos_rev"].append(kl(1, 0))
            res["kl_nl"].append(kl(2, 3)); res["kl_nl_rev"].append(kl(3, 2))
            res["dnll_eos"].append((nll[1] - nll[0]).cpu().numpy())
            res["dnll_nl"].append((nll[3] - nll[2]).cpu().numpy())
            res["flip_eos"].append((top1[0] != top1[1]).cpu().numpy())
            res["flip_nl"].append((top1[2] != top1[3]).cpu().numpy())
        else:
            res["kl_ceil"].append(kl(0, 1)); res["kl_ceil_rev"].append(kl(1, 0))
            res["dnll_ceil"].append((nll[1] - nll[0]).cpu().numpy())
            res["flip_ceil"].append((top1[0] != top1[1]).cpu().numpy())

    buf, next_unit = [], 0
    with torch.no_grad():
        for b0 in range(0, len(seqs), BS):
            batch = torch.stack(seqs[b0:b0 + BS]).to(DEVICE)
            logits = model(batch)
            for i in range(len(batch)):
                s = seq_start[b0 + i]
                buf.append(torch.log_softmax(logits[i, s:s + D].float(), -1))
            del logits
            while next_unit < len(units) and len(buf) >= units[next_unit]["n"]:
                u = units[next_unit]
                process(u, torch.stack(buf[:u["n"]]))
                del buf[:u["n"]]
                next_unit += 1
            if (b0 // BS) % 100 == 0:
                print(f"  batch {b0 // BS}/{(len(seqs) - 1) // BS + 1}")
    assert next_unit == len(units) and not buf

    out = {k: np.array(v) for k, v in res.items()}
    out["ps"] = np.array([events[i][1] for i in keep])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"saved {CACHE}")


if not CACHE.exists():
    compute()
z = np.load(CACHE)
ds = np.arange(1, D + 1)
n_ev, n_ce = len(z["kl_eos"]), len(z["kl_ceil"])

print(f"\n{n_ev} EOS events, {n_ce} mid-doc ceiling swaps; offsets d = 1..{D}")
print("\n   d | mean KL:  EOS      \\n     mid-doc |  flip rate: EOS    \\n  "
      "mid-doc | mean|dNLL|: EOS    \\n")
for d in [1, 2, 4, 8, 16, 32, 64]:
    i = d - 1
    print(f"  {d:2d} |   {z['kl_eos'][:, i].mean():8.4f} {z['kl_nl'][:, i].mean():7.3f}"
          f" {z['kl_ceil'][:, i].mean():7.3f} |     {z['flip_eos'][:, i].mean():6.4f}"
          f" {z['flip_nl'][:, i].mean():6.3f} {z['flip_ceil'][:, i].mean():6.3f} |"
          f"     {np.abs(z['dnll_eos'][:, i]).mean():7.4f} {np.abs(z['dnll_nl'][:, i]).mean():6.3f}")
m_eos, m_nl = z["kl_eos"].mean(), z["kl_nl"].mean()
print(f"\nmean over all d: KL_eos = {m_eos:.4f}, KL_nl = {m_nl:.4f}, "
      f"KL_ceil = {z['kl_ceil'].mean():.4f}; suppression KL_nl/KL_eos = {m_nl / m_eos:.1f}x")
print(f"reverse-KL means (swap||orig): eos {z['kl_eos_rev'].mean():.4f}, "
      f"nl {z['kl_nl_rev'].mean():.4f}, ceil {z['kl_ceil_rev'].mean():.4f}")
print(f"signed mean dNLL (swap - orig): eos {z['dnll_eos'].mean():+.4f}, "
      f"nl {z['dnll_nl'].mean():+.4f}, ceil {z['dnll_ceil'].mean():+.4f}")

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
(ax_m, ax_q), (ax_d, ax_f) = axes
curves = [("mid-doc swap (no EOS, suffix continues context)", "kl_ceil", C_CEIL),
          ("swap across '\\n' (same boundary, no EOS)", "kl_nl", C_NL),
          ("swap across <|endoftext|>", "kl_eos", C_EOS)]

for label, key, c in curves:
    ax_m.plot(ds, z[key].mean(0), color=c, lw=1.8, label=label)
ax_m.set_yscale("log")
ax_m.set_xlabel("offset d after boundary (tokens)")
ax_m.set_ylabel("mean KL(orig ‖ swapped)  [nats]")
ax_m.set_title("mean prediction change from swapping the pre-boundary context", fontsize=11)
ax_m.legend(fontsize=8, frameon=False)

for label, key, c in curves:
    med = np.median(z[key], 0)
    lo, hi = np.percentile(z[key], [25, 75], axis=0)
    ax_q.plot(ds, med, color=c, lw=1.8, label=label)
    ax_q.fill_between(ds, lo, hi, color=c, alpha=0.18, lw=0)
ax_q.set_yscale("log")
ax_q.set_xlabel("offset d after boundary (tokens)")
ax_q.set_ylabel("median KL, IQR band  [nats]")
ax_q.set_title("median and interquartile range (heavy-tail check)", fontsize=11)
ax_q.legend(fontsize=8, frameon=False)

for label, key, c in [(l, k.replace("kl", "dnll"), c) for l, k, c in curves]:
    ax_d.plot(ds, np.abs(z[key]).mean(0), color=c, lw=1.8, label=label)
ax_d.set_yscale("log")
ax_d.set_xlabel("offset d after boundary (tokens)")
ax_d.set_ylabel("mean |Δ log P(true token)|  [nats]")
ax_d.set_title("true-token log-prob change", fontsize=11)
ax_d.legend(fontsize=8, frameon=False)

floor = 1 / (2 * n_ev)
for label, key, c in [(l, k.replace("kl", "flip"), c) for l, k, c in curves]:
    ax_f.plot(ds, np.maximum(z[key].mean(0), floor), color=c, lw=1.8, label=label)
ax_f.axhline(floor, color="#aaaaaa", lw=0.8, ls=":")
ax_f.text(D, floor * 1.15, f"floor 1/2n", fontsize=7, ha="right", color="#888888")
ax_f.set_yscale("log")
ax_f.set_xlabel("offset d after boundary (tokens)")
ax_f.set_ylabel("top-1 flip rate")
ax_f.set_title("how often the argmax prediction changes", fontsize=11)
ax_f.legend(fontsize=8, frameon=False)

for ax in axes.flat:
    style(ax)
fig.suptitle("pile_4l — context-swap test: do predictions after the boundary depend on "
             f"what stood before it?  ({n_ev} EOS events, {n_ce} mid-doc swaps)", y=0.995)
fig.tight_layout()
fig.savefig(HERE.parent / "exp1_swap.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'exp1_swap.png'}")
