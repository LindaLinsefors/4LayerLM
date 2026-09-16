"""Tables + figure from the Modal per-position CI stats (pos0_ci.npz) for the
non-q/k matrices of layers 0-1: how many components of each matrix are
early-position-locked or EOS-locked in causal importance, and the CI firing
profiles of the headline mechanism components.

Definitions (per component, over 4000 rows = 2.048M tokens, 1412 mid-seq EOS):
  fire            = CI (lower_leaky) > 0.1
  early-locked    = >= 50 fires and > 50% of them at chunk position < 8
  EOS-locked      = >= 50 fires and > 50% of them at mid-sequence EOS positions

Output: endoftext-pos0/pos0_ci.png + printed tables (consumed by
pos0_components.md).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
d = np.load(HERE / "cache" / "pos0_ci.npz")
n_rows, n_eos = int(d["n_rows"]), int(d["n_eos"])
MATS = [f"h.{l}.{m}" for l in (0, 1)
        for m in ("attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]

counts = {}
for m in MATS:
    Fp, Fe = d[f"{m}|Fp"], d[f"{m}|Fe"]
    F = Fp.sum(1)
    with np.errstate(invalid="ignore"):
        efrac, eosfrac = Fp[:, :8].sum(1) / F, Fe / F
    counts[m] = (int((F > 0).sum()),
                 int(((F >= 50) & (efrac > 0.5)).sum()),
                 int(((F >= 50) & (eosfrac > 0.5)).sum()))
    print(f"{m}: fired {counts[m][0]:5d} / {len(F)}   early-locked {counts[m][1]:4d}   "
          f"EOS-locked {counts[m][2]:3d}")

HEADLINE = [
    ("h.1.mlp.down_proj", 1320, "dp2:1320 — the boundary writer (67% of u)"),
    ("h.1.mlp.down_proj", 828, "dp2:828 — EOS-side writer"),
    ("h.1.mlp.c_fc", 1689, "cfc2:1689 — trigger reader"),
    ("h.1.attn.o_proj", 292, "o2:292 — 'early in chunk' flag"),
    ("h.1.attn.o_proj", 180, "o2:180 — pos-0 attn-2 output"),
    ("h.0.mlp.down_proj", 216, "dp1:216 — MLP-1 amplifier"),
    ("h.0.attn.o_proj", 675, "o1:675 — pos-0 attn-1 output"),
]

fig, (ax_cnt, ax_prof) = plt.subplots(1, 2, figsize=(13.5, 5))

x = np.arange(len(MATS))
ax_cnt.bar(x - 0.25, [counts[m][0] for m in MATS], 0.25, label="fired at all",
           color="tab:gray")
ax_cnt.bar(x, [counts[m][1] for m in MATS], 0.25, label="early-locked (>50% of fires at pos < 8)",
           color="tab:orange")
ax_cnt.bar(x + 0.25, [counts[m][2] for m in MATS], 0.25, label="EOS-locked",
           color="tab:red")
ax_cnt.set_xticks(x, [m.replace("attn.", "").replace("mlp.", "").replace("_proj", "")
                      for m in MATS], rotation=45, ha="right")
ax_cnt.set_yscale("log")
ax_cnt.set_ylabel("components")
ax_cnt.set_title("position-locked components per matrix (CI > 0.1 fires)", fontsize=11)
ax_cnt.legend(fontsize=8, frameon=False)
sec = ax_cnt.secondary_xaxis("top")
sec.set_xticks([1.5, 5.5], ["layer 0 (attn 1 / MLP 1)", "layer 1 (attn 2 / MLP 2)"])
sec.tick_params(length=0)

ps = np.arange(32)
for m, c, lab in HEADLINE:
    rate = d[f"{m}|Fp"][c, :32] / n_rows
    ax_prof.plot(ps, rate, "o-", ms=3, lw=1.2, label=lab)
ax_prof.set_xlabel("chunk position p")
ax_prof.set_ylabel("CI fire rate (share of rows with CI > 0.1)")
ax_prof.set_title("CI firing profiles of the headline mechanism components", fontsize=11)
ax_prof.legend(fontsize=8, frameon=False)

for ax in (ax_cnt, ax_prof):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — CI position dependence, layers 0–1 v/o/MLP components "
             f"({n_rows} Pile rows; {n_eos} mid-seq EOS positions)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_ci.png", dpi=150, bbox_inches="tight")
print(f"saved {HERE.parent / 'pos0_ci.png'}")
