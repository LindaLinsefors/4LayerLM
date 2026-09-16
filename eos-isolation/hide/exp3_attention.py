"""Experiment 3 — mechanism: where does attention from post-EOS queries go?

Part A (natural rows). For every EOS event (boundary p), recompute per-head
attention (the model runs flash attention, so A is never materialized) and,
for query positions q = p+d, d = 1..64, split the attention mass by key range:

    key 0            (position-0 sink)
    keys 1..p-1      (pre-EOS content -- the "leak" channel)
    key p            (the EOS itself -- known sink site)
    keys p+1..q      (own document)

plus the value-weighted share  s_R = sum_{j in R} A_qj w_j / sum_j A_qj w_j,
w_j = ||W_O^h v_j||  (share of the head's output norm sourced from range R;
sink keys have suppressed values, so mass overstates their contribution).
Control: EOS-free rows with pseudo-boundaries drawn from the same p
distribution ("content more than d tokens back" without any boundary).

Part B (repeat sequences, the exp-2 stimuli). For [prefix, sep, D] with
prefix in {D, D'} and sep in {EOS, '\n'}, measure per head the mass from
second-copy queries onto prefix keys, and two specific keys for the query
holding D_i: the key holding D_i (duplicate-token mass, lag 65) and the key
holding D_{i+1} (induction-target mass, lag 64). The copies sit at a fixed
lag, so a purely positional head could hit these keys too — the same-vs-other
delta is the content-gated (true match) part. Heads with a large delta that
survives sep = EOS are the leak channel of experiment 2.

Cache: hide/cache/exp3.npz. Figures: ../exp3_attention.png, ../exp3_heads.png,
../exp3_leak_heads.png.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (C_DOC, C_EOS, C_KEY0, C_PRE, D, EOS_ID, HERE, NL_ID,
                    ROOT, attn_probs, find_events, load_rows, style)

import sys
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 8
N_CTRL = 500
M_DOCS = 400          # docs for part B
CACHE = HERE / "cache" / "exp3.npz"
BCONDS = ["same_eos", "same_nl", "other_eos", "other_nl"]


def compute():
    rows = load_rows()
    events = find_events(rows)
    rng = np.random.default_rng(0)
    eos_free = np.nonzero(~(rows == EOS_ID).any(1).numpy())[0]
    ctrl_rows = rng.choice(eos_free, size=N_CTRL, replace=False)
    ctrl_ps = rng.choice(np.array([p for _, p in events]), size=N_CTRL)

    items = {}  # row -> list of (p, cond)  cond 0 = EOS event, 1 = control
    for r, p in events:
        items.setdefault(int(r), []).append((p, 0))
    for r, p in zip(ctrl_rows, ctrl_ps):
        items.setdefault(int(r), []).append((int(p), 1))
    row_list = sorted(items)

    model, _, _ = load_pile_4l()
    model = model.to(DEVICE)
    L, H = len(model.h), model.h[0].attn.n_head

    grams = []
    for l in range(L):
        WO = model.h[l].attn.o_proj.weight.detach()
        hd = model.h[l].attn.head_dim
        grams.append(torch.stack([
            WO[:, h * hd:(h + 1) * hd].T @ WO[:, h * hd:(h + 1) * hd]
            for h in range(H)]).to(DEVICE))

    caps = {}
    hooks = [model.h[l].attn.register_forward_pre_hook(
        lambda _m, inp, l=l: caps.__setitem__(l, inp[0])) for l in range(L)]

    keys = ["m0", "m_pre", "m_bnd", "m_doc", "s0", "s_pre", "s_bnd"]
    sums = np.zeros((2, len(keys), L, H, D))
    cnt = np.zeros(2)

    def value_norms(l, x):
        attn = model.h[l].attn
        B, T_, _ = x.shape
        v = attn.v_proj(x).view(B, T_, attn.n_key_value_heads,
                                attn.head_dim).transpose(1, 2)
        if attn.repeat_kv_heads > 1:
            v = v.repeat_interleave(attn.repeat_kv_heads, dim=1)
        return torch.sqrt(torch.einsum("bhtc,hcd,bhtd->bht", v, grams[l], v))

    with torch.no_grad():
        for b0 in range(0, len(row_list), BS):
            rids = row_list[b0:b0 + BS]
            batch = torch.stack([rows[r] for r in rids]).to(DEVICE)
            model(batch)
            for l in range(L):
                A = attn_probs(model.h[l].attn, caps[l]).float()
                wn = value_norms(l, caps[l]).float()
                for b, r in enumerate(rids):
                    for p, cond in items[r]:
                        sl = A[b, :, p + 1:p + 1 + D, :]        # (H, D, T)
                        w = wn[b]                               # (H, T)
                        m0 = sl[:, :, 0]
                        m_pre = sl[:, :, 1:p].sum(-1)
                        m_bnd = sl[:, :, p]
                        m_doc = 1 - m0 - m_pre - m_bnd
                        den = (sl * w[:, None, :]).sum(-1)
                        s_pre = (sl[:, :, 1:p] * w[:, None, 1:p]).sum(-1) / den
                        s_bnd = sl[:, :, p] * w[:, p:p + 1] / den
                        s0 = sl[:, :, 0] * w[:, 0:1] / den
                        for ki, t in enumerate([m0, m_pre, m_bnd, m_doc,
                                                s0, s_pre, s_bnd]):
                            sums[cond, ki, l] += t.cpu().numpy()
                        if l == 0:
                            cnt[cond] += 1
            if (b0 // BS) % 25 == 0:
                print(f"  part A batch {b0 // BS}/{(len(row_list) - 1) // BS + 1}")
    out = {f"{k}_{c}": sums[c, ki] / cnt[c]
           for ki, k in enumerate(keys) for c in (0, 1)}
    out["cnt"] = cnt

    # ---- part B: repeat sequences ----
    docs = torch.stack([rows[r, p + 1:p + 1 + D] for r, p in events])
    n = len(events)
    other = np.arange(n)
    for i in range(n):
        j = (i + 37) % n
        while events[j][0] == events[i][0]:
            j = (j + 1) % n
        other[i] = j
    sel = np.linspace(0, n - 1, M_DOCS).astype(int)
    sep = {"eos": torch.full((len(sel), 1), EOS_ID, dtype=torch.long),
           "nl": torch.full((len(sel), 1), NL_ID, dtype=torch.long)}
    seqs = {f"{pre}_{s}": torch.cat(
        [docs[sel] if pre == "same" else docs[other[sel]], sep[s], docs[sel]], 1)
        for pre in ("same", "other") for s in ("eos", "nl")}

    i_idx = torch.arange(D - 1)                  # copy index i = 0..62
    for cond in BCONDS:
        pre_mass = np.zeros((L, H))
        sep_mass = np.zeros((L, H))
        tgt_mass = np.zeros((L, H))              # query holds D_i, key holds D_{i+1}
        dup_mass = np.zeros((L, H))              # query holds D_i, key holds D_i
        with torch.no_grad():
            for b0 in range(0, len(sel), 32):
                batch = seqs[cond][b0:b0 + 32].to(DEVICE)
                model(batch)
                for l in range(L):
                    A = attn_probs(model.h[l].attn, caps[l]).float()
                    pre_mass[l] += A[:, :, D + 1:, :D].sum(-1).mean(-1).sum(0).cpu().numpy()
                    sep_mass[l] += A[:, :, D + 1:, D].mean(-1).sum(0).cpu().numpy()
                    tgt_mass[l] += A[:, :, D + 1 + i_idx, i_idx + 1].mean(-1).sum(0).cpu().numpy()
                    dup_mass[l] += A[:, :, D + 1 + i_idx, i_idx].mean(-1).sum(0).cpu().numpy()
        out[f"b_pre_{cond}"] = pre_mass / len(sel)
        out[f"b_sep_{cond}"] = sep_mass / len(sel)
        out[f"b_tgt_{cond}"] = tgt_mass / len(sel)
        out[f"b_dup_{cond}"] = dup_mass / len(sel)
        print(f"  part B {cond} done")
    for h in hooks:
        h.remove()

    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"saved {CACHE}")


if not CACHE.exists():
    compute()
z = np.load(CACHE)
L, H = z["m0_0"].shape[:2]
ds = np.arange(1, D + 1)
n_ev, n_ct = int(z["cnt"][0]), int(z["cnt"][1])

print(f"\npart A: {n_ev} EOS events, {n_ct} control pseudo-boundaries")
print("\nmean attention mass from post-boundary queries (d = 1..64), heads averaged:")
print("        pre-boundary content      boundary key         key 0")
print("        EOS rows   control        EOS     control      EOS    control")
for l in range(L):
    print(f"  L{l}:    {z['m_pre_0'][l].mean():.4f}    {z['m_pre_1'][l].mean():.4f}"
          f"        {z['m_bnd_0'][l].mean():.4f}  {z['m_bnd_1'][l].mean():.4f}"
          f"      {z['m0_0'][l].mean():.4f}  {z['m0_1'][l].mean():.4f}")
print("\nvalue-weighted share of output sourced from pre-boundary content:")
for l in range(L):
    print(f"  L{l}:    EOS rows {z['s_pre_0'][l].mean():.4f}   "
          f"control {z['s_pre_1'][l].mean():.4f}")

d_tgt = z["b_tgt_same_eos"] - z["b_tgt_other_eos"]
d_tgt_nl = z["b_tgt_same_nl"] - z["b_tgt_other_nl"]
d_dup = z["b_dup_same_eos"] - z["b_dup_other_eos"]
d_dup_nl = z["b_dup_same_nl"] - z["b_dup_other_nl"]
flat = [(l, h) for l in range(L) for h in range(H)]
for name, dd, dd_nl, same_key in [
        ("induction-target mass (query holds D_i -> key holds D_(i+1))",
         d_tgt, d_tgt_nl, "b_tgt_same_eos"),
        ("duplicate-token mass (query holds D_i -> key holds D_i)",
         d_dup, d_dup_nl, "b_dup_same_eos")]:
    top = sorted(flat, key=lambda lh: -dd[lh])[:5]
    print(f"\npart B: {name}:")
    print("  head | same+EOS  delta(same-other) || delta with '\\n' | EOS/nl delta ratio")
    for l, h in top:
        r = dd[l, h] / dd_nl[l, h] if dd_nl[l, h] > 1e-4 else np.nan
        print(f"  L{l}h{h} |   {z[same_key][l, h]:.3f}       {dd[l, h]:+.3f}"
              f"         ||     {dd_nl[l, h]:+.3f}      |   {r:.2f}")

# ---- figure: per-layer mass + value-weighted share vs d ----
fig, axes = plt.subplots(2, L, figsize=(4.0 * L, 7.2), sharex=True)
for l in range(L):
    ax = axes[0, l]
    ax.plot(ds, z["m_pre_0"][l].mean(0), color=C_PRE, lw=1.8, label="pre-EOS content")
    ax.plot(ds, z["m_pre_1"][l].mean(0), color=C_PRE, lw=1.4, ls="--",
            label="same range, no EOS (control)")
    ax.plot(ds, z["m_bnd_0"][l].mean(0), color=C_EOS, lw=1.8, label="the EOS key")
    ax.plot(ds, z["m_bnd_1"][l].mean(0), color=C_EOS, lw=1.4, ls="--",
            label="boundary key (control)")
    ax.plot(ds, z["m0_0"][l].mean(0), color=C_KEY0, lw=1.8, label="key 0 (EOS rows)")
    ax.plot(ds, z["m0_1"][l].mean(0), color=C_KEY0, lw=1.4, ls="--",
            label="key 0 (control)")
    ax.plot(ds, z["m_doc_0"][l].mean(0), color=C_DOC, lw=1.8, label="own document")
    ax.set_yscale("log")
    ax.set_title(f"L{l} — attention mass", fontsize=10)
    if l == 0:
        ax.set_ylabel("mean mass (heads averaged)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axes[1, l]
    ax.plot(ds, z["s_pre_0"][l].mean(0), color=C_PRE, lw=1.8, label="pre-EOS content")
    ax.plot(ds, z["s_pre_1"][l].mean(0), color=C_PRE, lw=1.4, ls="--", label="control")
    ax.plot(ds, z["s_bnd_0"][l].mean(0), color=C_EOS, lw=1.8, label="the EOS key")
    ax.plot(ds, z["s0_0"][l].mean(0), color=C_KEY0, lw=1.8, label="key 0")
    ax.set_yscale("log")
    ax.set_xlabel("offset d after boundary")
    ax.set_title(f"L{l} — value-weighted share", fontsize=10)
    if l == 0:
        ax.set_ylabel("share of head output norm")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
for ax in axes.flat:
    style(ax)
fig.suptitle("pile_4l — where post-boundary queries attend: mass (top) and value-weighted "
             f"output share (bottom); {n_ev} EOS events vs {n_ct} no-EOS controls", y=0.995)
fig.tight_layout()
fig.savefig(HERE.parent / "exp3_attention.png", dpi=150, bbox_inches="tight")

# ---- figure: per-head pre-boundary mass ----
fig, axes = plt.subplots(L, H, figsize=(2.1 * H, 2.0 * L), sharex=True, sharey=True)
for l in range(L):
    for h in range(H):
        ax = axes[l, h]
        ax.plot(ds, z["m_pre_0"][l, h], color=C_PRE, lw=1.5)
        ax.plot(ds, z["m_pre_1"][l, h], color=C_PRE, lw=1.1, ls="--")
        ax.plot(ds, z["m_bnd_0"][l, h], color=C_EOS, lw=1.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.text(0.05, 0.05, f"L{l}h{h}", transform=ax.transAxes, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if l == L - 1:
            ax.set_xlabel("d", fontsize=8)
fig.suptitle("per-head mass on pre-EOS content (solid purple), same range in no-EOS "
             "controls (dashed), and on the EOS key (vermillion)", y=0.998)
fig.tight_layout()
fig.savefig(HERE.parent / "exp3_heads.png", dpi=150, bbox_inches="tight")

# ---- figure: part B induction / duplicate-token heads ----
fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
mats = [[(z["b_tgt_same_eos"], "induction-target mass, D + EOS + D"),
         (d_tgt, "Δ target mass (same − other), EOS"),
         (d_tgt_nl, "Δ target mass (same − other), '\\n'")],
        [(z["b_dup_same_eos"], "duplicate-token mass, D + EOS + D"),
         (d_dup, "Δ duplicate mass (same − other), EOS"),
         (d_dup_nl, "Δ duplicate mass (same − other), '\\n'")]]
vmax = max(m.max() for row in mats for m, _ in row)
for r in range(2):
    for ax, (m, title) in zip(axes[r], mats[r]):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        for l in range(L):
            for h in range(H):
                ax.text(h, l, f"{m[l, h]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if m[l, h] < vmax * 0.6 else "k")
        ax.set_xticks(range(H), [f"h{h}" for h in range(H)])
        ax.set_yticks(range(L), [f"L{l}" for l in range(L)])
        ax.set_title(title, fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.85)
fig.suptitle("heads matching the earlier copy: successor key (top, lag 64) and "
             "same-token key (bottom, lag 65); Δ(same − other) isolates the "
             "content-gated part", y=0.98)
fig.savefig(HERE.parent / "exp3_leak_heads.png", dpi=150, bbox_inches="tight")
print(f"saved figures to {HERE.parent}")
