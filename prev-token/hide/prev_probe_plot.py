"""Figures + tables for the previous-token linear-probe analysis (sink seed 45).

Reads hide/cache/prev_probe.npz (from prev_probe_modal.py), writes
  prev-token/prev_probe.png        test top-1 / top-5 / NLL per stage, both variants
  prev-token/prev_probe_pos.png    test top-1 vs position for selected stages
and prints the markdown summary table for the report.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent  # scripts in hide/ write human-readable outputs one level up
data = np.load(HERE / "cache" / "prev_probe.npz")

STAGES = ["emb"] + [f"{k}{l}" for l in range(1, 5) for k in ("attn", "mlp")] + ["ln_f"]
LABELS = ["emb"] + [f"{k} {l}" for l in range(1, 5) for k in ("attn", "mlp")] + ["ln_f"]
BLUE, ORANGE = "#0072B2", "#E69F00"  # Okabe-Ito

get = lambda s, v, m: float(data[f"{s}|{v}|{m}"])

# --- summary figure: top-1, top-5, NLL per stage ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
x = np.arange(len(STAGES))
for metric, ax, title in [("test_top1", axes[0], "held-out top-1 accuracy"),
                          ("test_top5", axes[1], "held-out top-5 accuracy"),
                          ("test_nll", axes[2], "held-out NLL (nats)")]:
    for variant, color, name in [("lin", BLUE, "linear"),
                                 ("norm", ORANGE, "RMSNorm + linear")]:
        ax.plot(x, [get(s, variant, metric) for s in STAGES],
                "o-", color=color, lw=2, ms=5, label=name)
    ax.set_title(title)
    ax.set_xticks(x, LABELS, rotation=45, ha="right")
    ax.grid(alpha=0.25)
    if metric != "test_nll":
        ax.set_ylim(0, 1.02)
axes[0].legend(frameon=False)
fig.suptitle("Previous-token probe on sink seed 45's residual stream "
             "(trained unembedding, 1,600/400 train/test rows)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "prev_probe.png", dpi=150, bbox_inches="tight")
print("saved", OUT / "prev_probe.png")

# --- accuracy vs position for the interesting stages ---
fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.viridis(np.linspace(0, 0.9, 4))
for stage, c in zip(["emb", "attn1", "mlp1", "attn2"], colors):
    acc = data[f"{stage}|norm|pos_acc"]
    pos = np.arange(1, len(acc) + 1)
    # smooth with a 9-position moving average for readability
    kern = np.ones(9) / 9
    ax.plot(pos[4:-4], np.convolve(acc, kern, "valid"), color=c, lw=1.5,
            label=f"after {stage}")
ax.set_xlabel("position in chunk (predicting token at position − 1)")
ax.set_ylabel("test top-1 accuracy")
ax.set_title("Previous-token probe accuracy vs position (RMSNorm + linear variant)")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "prev_probe_pos.png", dpi=150, bbox_inches="tight")
print("saved", OUT / "prev_probe_pos.png")

# --- markdown table for the report ---
print("\n| stage | top-1 (lin) | top-1 (norm) | top-5 (lin) | top-5 (norm) "
      "| NLL (lin) | NLL (norm) | top-1 mid-train (lin) | train top-1 (lin) |")
print("|---|---|---|---|---|---|---|---|---|")
for s, lab in zip(STAGES, LABELS):
    print(f"| {lab} | {get(s,'lin','test_top1'):.3f} | {get(s,'norm','test_top1'):.3f} "
          f"| {get(s,'lin','test_top5'):.3f} | {get(s,'norm','test_top5'):.3f} "
          f"| {get(s,'lin','test_nll'):.2f} | {get(s,'norm','test_nll'):.2f} "
          f"| {get(s,'lin','mid_top1'):.3f} | {get(s,'lin','train_top1'):.3f} |")
