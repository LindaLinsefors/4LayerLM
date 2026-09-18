"""Figures for the copy-prediction experiment (reads cache/copy_predict.npz).

Writes (one level up, human-readable):
  copy_nll.png       - mean NLL vs offset into the second copy, panel per model,
                       solid = repeated (colored by gap, light->dark blue),
                       dashed gray = control (novel text; gaps overlap, so the
                       largest-gap control is drawn as the reference)
  copy_accuracy.png  - same layout, top-1 accuracy
  copy_summary.png   - copy benefit (nats) and top-1 accuracy vs gap, per model
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent  # human-readable outputs one level up from hide/
d = np.load(HERE / "cache" / "copy_predict.npz")

MODELS = ["pile_4l", "sink45", "sink46", "simple_2l"]
TITLES = {"pile_4l": "pile_4l (t-9d2b8f02)", "sink45": "sink seed 45 (t-87f91319)",
          "sink46": "sink seed 46 (t-75f6c439)", "simple_2l": "simple_2l (gf6rbga0)"}
MODEL_C = {"pile_4l": "#0072B2", "sink45": "#E69F00", "sink46": "#009E73", "simple_2l": "#CC79A7"}
GAP_C = ["#b3cde3", "#6497b1", "#2a6f97", "#03254c"]  # light -> dark, one hue


def style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def per_offset(fig_name, key, ylabel, ylog):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=False)
    for ax, m in zip(axes.flat, MODELS):
        gaps = d[f"{m}|gaps"]; S = int(d[f"{m}|S"])
        offs = np.arange(1, S)
        for gi, g in enumerate(gaps):
            ax.plot(offs, d[f"{m}|rep|{key}|{g}"][:, 1:].mean(0), color=GAP_C[gi],
                    lw=2, label=f"gap {g}")
        gmax = gaps[-1]
        ax.plot(offs, d[f"{m}|ctl|{key}|{gmax}"][:, 1:].mean(0), color="#888888",
                lw=2, ls="--", label="control (novel)")
        if ylog:
            ax.set_yscale("log")
        ax.set_title(TITLES[m], fontsize=11)
        ax.set_xlabel("offset $j$ into second copy")
        ax.set_ylabel(ylabel)
        style(ax)
    axes[0, 0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle("Second occurrence of a context span: " + ylabel, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / fig_name, dpi=150)
    plt.close(fig)


per_offset("copy_nll.png", "nll", "mean NLL (nats)", ylog=True)
per_offset("copy_accuracy.png", "acc", "top-1 accuracy", ylog=False)

# summary: benefit + accuracy vs gap (offsets >= 8, past the induction ramp-up)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for m in MODELS:
    gaps = d[f"{m}|gaps"]
    ben = [d[f"{m}|ctl|nll|{g}"][:, 8:].mean() - d[f"{m}|rep|nll|{g}"][:, 8:].mean() for g in gaps]
    acc = [d[f"{m}|rep|acc|{g}"][:, 8:].mean() for g in gaps]
    lbl = TITLES[m].split(" (")[0]
    ax1.plot(gaps, ben, "o-", color=MODEL_C[m], lw=2, ms=6, label=lbl)
    ax2.plot(gaps, acc, "o-", color=MODEL_C[m], lw=2, ms=6, label=lbl)
    ax2.plot(gaps, [d[f"{m}|ctl|acc|{g}"][:, 8:].mean() for g in gaps], ":",
             color=MODEL_C[m], lw=1.5, alpha=0.7)
ax1.set_xlabel("gap $g$ between the two copies (tokens)")
ax1.set_ylabel("copy benefit (nats),  offsets $\\geq 8$")
ax1.set_title("NLL saved by having seen the span before")
ax2.set_xlabel("gap $g$ between the two copies (tokens)")
ax2.set_ylabel("top-1 accuracy,  offsets $\\geq 8$")
ax2.set_title("Copy accuracy (dotted: control baseline)")
for ax in (ax1, ax2):
    style(ax)
    ax.legend(fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "copy_summary.png", dpi=150)
plt.close(fig)
print("figures written to", OUT)
