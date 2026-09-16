"""Experiment 4 — is the position-0 sink machinery causally responsible for
the EOS firewall?

endoftext-pos0/ showed the pos-0 and mid-seq-EOS sinks share one massive-vector
direction (cos 0.962) and one dominant writer component (h.1.mlp.down_proj:1320,
CI 1.00 at both sites); 6 of the top-10 writers by |pos-0 contribution| are
also in the top-10 by |EOS contribution|. If the firewall of experiments 1-3
runs on that shared machinery, surgically ablating the writers (subtract
sum_c outer(U_c, V_c) from the actual weight) should re-open the boundary:

    exp-1 metric: KL(orig || context-swapped) across EOS should rise
    exp-2 metric: leak fraction / cross-EOS copy rate should rise toward the
                  '\n' level
    exp-3 metric: L2/L3 attention parked on the EOS key should collapse, mass
                  on pre-EOS content should return toward the no-EOS control

Variants: abl_eos10 (top-10 by |EOS| write onto u, 94% of the EOS write),
abl_pos10 (top-10 by |pos-0|, the published set), rand10 (10 random *alive*
h.1.mlp.down_proj components, control), plus the unablated baseline (validates
this script's lite pipelines against the exp1/2/3 caches). Writer sets from
endoftext-pos0/hide/cache/pos0_components.npz; also records post-MLP-2 norms
at pos-0/EOS/bulk per variant (ablation kill-quality check).

Cache: hide/cache/exp4.npz. Figure: ../exp4_pos0_ablation.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import (C_EOS, D, EOS_ID, HERE, NL_ID, ROOT,
                    assign_partners, attn_probs, find_events, load_rows, style)

import sys
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODULE = "h.1.mlp.down_proj"
CACHE = HERE / "cache" / "exp4.npz"
BCONDS = ["same_eos", "same_nl", "other_eos", "other_nl", "none"]
N_ROWS3 = 400          # event rows for the attention measure
N_CTRL3 = 200
BULK_P = 300

C_VAR = {"baseline": C_EOS, "abl_eos10": "#0072B2",
         "abl_pos10": "#56B4E9", "rand10": "#777777"}


def compute():
    rows = load_rows()
    events = find_events(rows)
    rng = np.random.default_rng(0)
    partners = assign_partners(events, rng)
    keep = [i for i in range(len(events)) if partners[i] >= 0]

    pc_cache = np.load(ROOT / "endoftext-pos0/hide/cache/pos0_components.npz")
    top_pos = np.argsort(-np.abs(pc_cache[f"{MODULE}|pos0"]))[:10]
    top_eos = np.argsort(-np.abs(pc_cache[f"{MODULE}|eos"]))[:10]
    mean_ci = np.load(ROOT / "coci-heatmaps/hide/cache/mean_ci_pile_4l.npz")[MODULE]
    alive = np.nonzero(mean_ci > 1e-6)[0]
    alive = np.setdiff1d(alive, np.union1d(top_pos, top_eos))
    rand10 = np.random.default_rng(1).choice(alive, 10, replace=False)
    variants = {"baseline": np.array([], dtype=int), "abl_eos10": top_eos,
                "abl_pos10": top_pos, "rand10": rand10}
    print(f"top10 by |pos0|: {sorted(top_pos.tolist())}")
    print(f"top10 by |EOS| : {sorted(top_eos.tolist())}")
    print(f"overlap: {sorted(set(top_pos.tolist()) & set(top_eos.tolist()))}, "
          f"rand10: {sorted(rand10.tolist())}")

    model, pc, _ = load_pile_4l()
    model = model.to(DEVICE)
    L, H = len(model.h), model.h[0].attn.n_head
    Vm = pc.components[MODULE].V.to(DEVICE)
    Um = pc.components[MODULE].U.to(DEVICE)
    lin = model.h[1].mlp.down_proj

    def ablate(comps):
        if len(comps) == 0:
            return lambda: None
        delta = (Vm[:, comps] @ Um[comps, :]).T.to(lin.weight.dtype)
        lin.weight.data -= delta
        return lambda: lin.weight.data.add_(delta)

    # ---------- shared stimuli ----------
    # exp-1 lite: per event 4 sequences [orig, swapE, orig_nl, swap_nl]
    ev_seqs, ev_ps, ev_true = [], [], []
    for i in keep:
        r, p = events[i]
        row = rows[r]
        rj, pj = events[partners[i]]
        rep = rows[rj, pj - p:pj]
        orig_nl = row.clone(); orig_nl[p] = NL_ID
        ev_seqs.append(torch.stack([
            row, torch.cat([rep, torch.tensor([EOS_ID]), row[p + 1:]]),
            orig_nl, torch.cat([rep, torch.tensor([NL_ID]), row[p + 1:]])]))
        ev_ps.append(p)
        ev_true.append(row[p + 1:p + 1 + D])

    # exp-2 lite: repeat sequences
    docs = torch.stack([rows[r, p + 1:p + 1 + D] for r, p in events])
    n2 = len(events)
    other = np.arange(n2)
    for i in range(n2):
        j = (i + 37) % n2
        while events[j][0] == events[i][0]:
            j = (j + 1) % n2
        other[i] = j
    sep_e = torch.full((n2, 1), EOS_ID, dtype=torch.long)
    sep_n = torch.full((n2, 1), NL_ID, dtype=torch.long)
    seqs2 = {"same_eos": torch.cat([docs, sep_e, docs], 1),
             "same_nl": torch.cat([docs, sep_n, docs], 1),
             "other_eos": torch.cat([docs[other], sep_e, docs], 1),
             "other_nl": torch.cat([docs[other], sep_n, docs], 1),
             "none": torch.cat([sep_e, docs], 1)}

    # exp-3 lite rows: subset of event rows + EOS-free controls
    ev_rows = {}
    for r, p in events:
        ev_rows.setdefault(int(r), []).append(p)
    sub_rows = np.random.default_rng(5).choice(
        sorted(ev_rows), min(N_ROWS3, len(ev_rows)), replace=False)
    rng3 = np.random.default_rng(3)
    eos_free = np.nonzero(~(rows == EOS_ID).any(1).numpy())[0]
    ctrl_rows = rng3.choice(eos_free, N_CTRL3, replace=False)
    ctrl_ps = rng3.choice(np.array([p for _, p in events]), N_CTRL3)
    items3 = [(int(r), ev_rows[int(r)], 0) for r in sub_rows] + \
             [(int(r), [int(p)], 1) for r, p in zip(ctrl_rows, ctrl_ps)]

    def measure_exp1():
        kl_sum = np.zeros((2, D))       # [eos pair, nl pair]
        flip_sum = np.zeros((2, D))
        d_idx = torch.arange(D, device=DEVICE)
        with torch.no_grad():
            for i0 in range(0, len(ev_seqs), 2):
                group = ev_seqs[i0:i0 + 2]
                batch = torch.cat(group).to(DEVICE)
                logits = model(batch)
                for gi, seqs4 in enumerate(group):
                    p = ev_ps[i0 + gi]
                    lp = torch.log_softmax(
                        logits[4 * gi:4 * gi + 4, p:p + D].float(), -1)
                    for pair, (a, b) in enumerate([(0, 1), (2, 3)]):
                        kl = (lp[a].exp() * (lp[a] - lp[b])).sum(-1)
                        kl_sum[pair] += kl.cpu().numpy()
                        flip_sum[pair] += (lp[a].argmax(-1)
                                           != lp[b].argmax(-1)).cpu().numpy()
                del logits
        return kl_sum / len(ev_seqs), flip_sum / len(ev_seqs)

    def measure_exp2():
        nll = np.zeros((len(BCONDS), D))
        acc = np.zeros((len(BCONDS), D))
        with torch.no_grad():
            for ci, cond in enumerate(BCONDS):
                s = seqs2[cond]
                start = 0 if cond == "none" else D
                for b0 in range(0, n2, 16):
                    batch = s[b0:b0 + 16].to(DEVICE)
                    lp = torch.log_softmax(model(batch)[:, start:start + D].float(), -1)
                    true = docs[b0:b0 + 16].to(DEVICE)
                    nll[ci] += (-lp.gather(-1, true[:, :, None])[:, :, 0]
                                ).sum(0).cpu().numpy()
                    acc[ci] += (lp.argmax(-1) == true).float().sum(0).cpu().numpy()
        return nll / n2, acc / n2

    def measure_exp3():
        caps = {}
        hooks = [model.h[l].attn.register_forward_pre_hook(
            lambda _m, inp, l=l: caps.__setitem__(l, inp[0])) for l in range(L)]
        hooks.append(model.h[1].register_forward_hook(
            lambda _m, _i, out: caps.__setitem__("b1", out)))
        mass = np.zeros((2, 3, L, H))   # [cond, {pre,bnd,key0}, l, h], mean over d
        cnt = np.zeros(2)
        norms = {"pos0": [], "eos": [], "bulk": []}
        with torch.no_grad():
            for b0 in range(0, len(items3), 8):
                grp = items3[b0:b0 + 8]
                batch = torch.stack([rows[r] for r, _, _ in grp]).to(DEVICE)
                model(batch)
                b1 = caps["b1"].float()
                for b, (r, _, cond) in enumerate(grp):
                    if cond == 0:
                        norms["pos0"].append(float(b1[b, 0].norm()))
                        norms["bulk"].append(float(b1[b, BULK_P].norm()))
                        for p in ev_rows[r]:
                            norms["eos"].append(float(b1[b, p].norm()))
                for l in range(L):
                    A = attn_probs(model.h[l].attn, caps[l]).float()
                    for b, (r, ps, cond) in enumerate(grp):
                        for p in ps:
                            sl = A[b, :, p + 1:p + 1 + D, :]
                            mass[cond, 0, l] += sl[:, :, 1:p].sum(-1).mean(-1).cpu().numpy()
                            mass[cond, 1, l] += sl[:, :, p].mean(-1).cpu().numpy()
                            mass[cond, 2, l] += sl[:, :, 0].mean(-1).cpu().numpy()
                            if l == 0:
                                cnt[cond] += 1
        for h in hooks:
            h.remove()
        return (mass / cnt[:, None, None, None],
                {g: np.mean(v) for g, v in norms.items()})

    out = {}
    for name, comps in variants.items():
        print(f"\n=== variant {name} ===")
        restore = ablate(comps)
        kl, flips = measure_exp1()
        print(f"  exp1: mean KL eos {kl[0].mean():.4f}, nl {kl[1].mean():.4f}")
        nll, acc = measure_exp2()
        sl = slice(1, D)
        gap_e = nll[2, sl].mean() - nll[0, sl].mean()
        gap_n = nll[3, sl].mean() - nll[1, sl].mean()
        print(f"  exp2: leak {gap_e / gap_n:.3f}, copy same_eos {acc[0, sl].mean():.3f}"
              f" same_nl {acc[1, sl].mean():.3f}")
        mass, norms = measure_exp3()
        print(f"  exp3: L2 m_pre {mass[0, 0, 2].mean():.3f} (ctrl {mass[1, 0, 2].mean():.3f})"
              f", L2 m_bnd {mass[0, 1, 2].mean():.3f}; norms pos0 {norms['pos0']:.0f}"
              f" EOS {norms['eos']:.0f} bulk {norms['bulk']:.1f}")
        restore()
        out[f"{name}/kl"] = kl
        out[f"{name}/flips"] = flips
        out[f"{name}/nll"] = nll
        out[f"{name}/acc"] = acc
        out[f"{name}/mass"] = mass
        out[f"{name}/norms"] = np.array([norms["pos0"], norms["eos"], norms["bulk"]])
    out["sets"] = np.stack([top_eos, top_pos, rand10])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"\nsaved {CACHE}")


if not CACHE.exists():
    compute()
z = np.load(CACHE)
ds = np.arange(1, D + 1)
names = ["baseline", "abl_eos10", "abl_pos10", "rand10"]
labels = {"baseline": "baseline", "abl_eos10": "ablate top-10 EOS writers",
          "abl_pos10": "ablate top-10 pos-0 writers", "rand10": "ablate 10 random alive"}
sl = slice(1, D)

print("\nsummary (means; leak over d=2..64):")
print("  variant        KL_eos   KL_nl   leak   copy_eos  copy_nl  L2 m_pre  L2 m_bnd  EOS-norm")
for v in names:
    kl = z[f"{v}/kl"]; nll = z[f"{v}/nll"]; acc = z[f"{v}/acc"]; mass = z[f"{v}/mass"]
    leak = (nll[2, sl].mean() - nll[0, sl].mean()) / (nll[3, sl].mean() - nll[1, sl].mean())
    print(f"  {v:12s}  {kl[0].mean():.4f}  {kl[1].mean():.3f}   {leak:.3f}"
          f"    {acc[0, sl].mean():.3f}     {acc[1, sl].mean():.3f}"
          f"    {mass[0, 0, 2].mean():.3f}     {mass[0, 1, 2].mean():.3f}"
          f"     {z[f'{v}/norms'][1]:.0f}")

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
(ax_kl, ax_leak), (ax_pre, ax_bnd) = axes

for v in names:
    ax_kl.plot(ds, z[f"{v}/kl"][0], color=C_VAR[v], lw=1.8, label=labels[v])
ax_kl.plot(ds, z["baseline/kl"][1], color="k", lw=1.2, ls="--",
           label="baseline, swap across '\\n' (no-firewall ref)")
ax_kl.set_yscale("log")
ax_kl.set_xlabel("offset d after boundary")
ax_kl.set_ylabel("mean KL(orig ‖ swapped)  [nats]")
ax_kl.set_title("exp-1 metric: context-swap sensitivity across EOS", fontsize=11)
ax_kl.legend(fontsize=8, frameon=False)

x = np.arange(len(names))
leaks = [(z[f"{v}/nll"][2, sl].mean() - z[f"{v}/nll"][0, sl].mean())
         / (z[f"{v}/nll"][3, sl].mean() - z[f"{v}/nll"][1, sl].mean()) for v in names]
ax_leak.bar(x, leaks, 0.55, color=[C_VAR[v] for v in names])
for xi, lv in zip(x, leaks):
    ax_leak.text(xi, lv + 0.01, f"{lv:.2f}", ha="center", fontsize=9)
ax_leak.axhline(1.0, color="k", ls=":", lw=1)
ax_leak.text(0.02, 1.01, "no firewall", fontsize=8, transform=ax_leak.get_yaxis_transform())
ax_leak.set_xticks(x, [labels[v] for v in names], rotation=12, fontsize=8)
ax_leak.set_ylabel("leak fraction")
ax_leak.set_title("exp-2 metric: induction leak fraction", fontsize=11)

w = 0.2
for li, l in enumerate((2, 3)):
    for vi, v in enumerate(names):
        m = z[f"{v}/mass"]
        ax_pre.bar(li + (vi - 1.5) * w, m[0, 0, l].mean(), w, color=C_VAR[v],
                   label=labels[v] if li == 0 else None)
        ax_bnd.bar(li + (vi - 1.5) * w, m[0, 1, l].mean(), w, color=C_VAR[v],
                   label=labels[v] if li == 0 else None)
    ax_pre.plot([li - 2 * w, li + 2 * w],
                [z["baseline/mass"][1, 0, l].mean()] * 2, "k--", lw=1.2)
ax_pre.text(0.6, z["baseline/mass"][1, 0, 3].mean() * 1.03,
            "no-EOS control level (firewall off)", fontsize=8)
for ax, title in [(ax_pre, "mass on pre-EOS content"),
                  (ax_bnd, "mass parked on the EOS key")]:
    ax.set_xticks([0, 1], ["L2", "L3"])
    ax.set_title(f"exp-3 metric: {title} (queries d=1..64, heads avg)", fontsize=11)
    ax.set_ylabel("mean attention mass")
    ax.legend(fontsize=8, frameon=False)
for ax in axes.flat:
    style(ax)
fig.suptitle("pile_4l — ablating the massive-vector writer components "
             "(h.1.mlp.down_proj): does the EOS firewall survive?", y=0.995)
fig.tight_layout()
fig.savefig(HERE.parent / "exp4_pos0_ablation.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'exp4_pos0_ablation.png'}")
