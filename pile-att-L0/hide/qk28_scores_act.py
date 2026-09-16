"""Pre-softmax attention scores between q:28 and every alive L0 k-component,
vs query-key distance -- second pass: scaled by data activation strengths.

The actual score contribution of the component pair (q:28, k:c) when both are
active is

    score(c, D) = a_q a_c * sum_h (R_D uhat_q,h) . uhat_c,h / sqrt(d_head)

where a = ||U|| (V . x) is the signed component activation (the write is
a * uhat, uhat = U/||U||). We use each component's *typical* activation on
tokens where it is causally important: s = mean(a) over harvest.db activation-
example tokens with CI > 0.1 (signed mean; the product s_q s_c is gauge-
invariant, unlike the unit-vector map's row signs). The distance factor is the
cached unit-vector score map from qk28_scores_norm.py.

Sanity check included: harvest's component_activation is recomputed locally as
||U_c|| * (V_c . rms_1(wte[t])) -- L0 attention input is token-only.

Writes ../qk28_scores_act.png; caches strengths in cache/act_stats.npz.
"""

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

HARVEST = ROOT / "prev_paper" / "models" / "pile_4layer" / "additional-component-data" / "harvest.db"
NORM_CACHE = HERE / "cache" / "qk28_scores_normU.npz"
STATS_CACHE = HERE / "cache" / "act_stats.npz"
CI_THRESH = 0.1

d = np.load(NORM_CACHE)
k_ids, unit_scores = d["k_ids"], d["scores"]  # (126,), (126, 512)


def harvest_stats(keys: list[str]) -> dict[str, tuple[float, float, int, int]]:
    """Per component: (signed mean act, mean |act|, n CI-active tokens, an example token id)."""
    con = sqlite3.connect(f"file:{HARVEST}?mode=ro", uri=True)
    out = {}
    for key in keys:
        (blob,) = con.execute(
            "SELECT activation_examples FROM components WHERE component_key=?", (key,)
        ).fetchone()
        acts, tok = [], -1
        for ex in json.loads(blob):
            ci = ex["activations"]["causal_importance"]
            a = ex["activations"]["component_activation"]
            for i, c in enumerate(ci):
                if c > CI_THRESH:
                    acts.append(a[i])
                    tok = ex["token_ids"][i]
        acts = np.array(acts)
        out[key] = (acts.mean() if len(acts) else 0.0,
                    np.abs(acts).mean() if len(acts) else 0.0, len(acts), tok)
    con.close()
    return out


if STATS_CACHE.exists():
    st = np.load(STATS_CACHE)
    s_q, s_k_signed, s_k_abs, n_fire = st["s_q"], st["s_k_signed"], st["s_k_abs"], st["n_fire"]
else:
    keys = ["h.0.attn.q_proj:28"] + [f"h.0.attn.k_proj:{i}" for i in k_ids]
    stats = harvest_stats(keys)
    s_q = stats[keys[0]][0]
    s_k_signed = np.array([stats[k][0] for k in keys[1:]])
    s_k_abs = np.array([stats[k][1] for k in keys[1:]])
    n_fire = np.array([stats[k][2] for k in keys[1:]])

    # --- sanity check: harvest act == ||U|| (V . rms_1(wte[t])) on a firing token ---
    import torch
    from load import load_pile_4l

    model, pc, _ = load_pile_4l()
    wte = model.wte.weight.float()
    rms1 = model.h[0].rms_1
    for key in [keys[0], "h.0.attn.k_proj:465", "h.0.attn.k_proj:29"]:
        mod = key.rsplit(":", 1)[0]
        idx = int(key.rsplit(":", 1)[1])
        V, U = pc.components[mod].V.float(), pc.components[mod].U.float()
        tok = stats[key][3]
        with torch.no_grad():
            x = rms1(wte[tok:tok + 1])[0]
            a_local = (U[idx].norm() * (V[:, idx] @ x)).item()
        # harvest act on that same token (any occurrence -- token-only at L0)
        con = sqlite3.connect(f"file:{HARVEST}?mode=ro", uri=True)
        (blob,) = con.execute(
            "SELECT activation_examples FROM components WHERE component_key=?", (key,)
        ).fetchone()
        con.close()
        a_harvest = next(
            ex["activations"]["component_activation"][i]
            for ex in json.loads(blob)
            for i, t in enumerate(ex["token_ids"])
            if t == tok and ex["activations"]["causal_importance"][i] > CI_THRESH
        )
        print(f"sanity {key} tok {tok}: local {a_local:+.3f}  harvest {a_harvest:+.3f}")

    STATS_CACHE.parent.mkdir(exist_ok=True)
    np.savez(STATS_CACHE, s_q=s_q, s_k_signed=s_k_signed, s_k_abs=s_k_abs, n_fire=n_fire)

print(f"q:28 typical signed activation on CI>{CI_THRESH} tokens: {s_q:+.2f}")
cancel = np.abs(s_k_signed) / np.maximum(s_k_abs, 1e-9)
print(f"k comps: |signed mean|/mean|a| median {np.median(cancel):.2f}; "
      f"{(cancel < 0.5).sum()} of {len(k_ids)} below 0.5 (mixed-sign activations)")

scores = s_q * s_k_signed[:, None] * unit_scores  # (126, 512), actual logit units

n = len(k_ids)
vmax = np.abs(scores).max()
print(f"max |score| = {vmax:.2f} logits")

fig, ax = plt.subplots(figsize=(12, 15))
im = ax.imshow(scores, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               extent=(-0.5, 512 - 0.5, n - 0.5, -0.5), interpolation="nearest")
ax.set_yticks(np.arange(n))
ax.set_yticklabels(k_ids, fontsize=4.5)
ax.set_xlabel("distance D = query pos $-$ key pos (tokens)")
ax.set_ylabel("alive k component (id order)")
ax.set_title("Typical pre-softmax score contribution  $s_{q28}\\, s_c \\sum_h (R_D \\hat u_{q,h})"
             "\\cdot\\hat u_{c,h}/\\sqrt{128}$\n"
             f"$s$ = signed mean component activation on CI > {CI_THRESH} tokens (harvest.db)",
             fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("score (logit units)")
fig.tight_layout()
out = HERE.parent / "qk28_scores_act.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
