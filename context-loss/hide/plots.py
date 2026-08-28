"""Analysis + plots for the context-loss results (run compute.py first).

Every token's context splits by whether the current document's start is visible:
  dist == -1 : no EOS in the context -- a mid-document token whose visible
               context is entirely same-document but TRUNCATED at the row start:
               exactly n = t+1 usable context tokens. The cleanest
               "loss vs context length" measurement.
  dist == d  : the document started inside the row -- the model sees the whole
               document so far; d measures POSITION in the document, not
               truncation.

Produces (in context-loss/):
  loss_vs_position.png     -- mean loss vs absolute context length (as trained),
                              both models, with saturating power-law fits
  truncation_vs_position.png -- per model: truncated-context curve vs
                              document-position curve (the decomposition above)
  out_of_doc_context.png   -- per model: document-position curve, banded by the
                              amount of unrelated (out-of-document) prefix
  fit_stats.txt            -- fit parameters and key numbers
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

HERE = Path(__file__).parent
MODELS = ["pile_4l", "simple_2l"]

# --- palette (validated reference palette, light mode) ---
BLUE, AQUA = "#2a78d6", "#1baf7a"
ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#104281"]  # ordinal ramp, light->dark
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
MODEL_COLOR = {"pile_4l": BLUE, "simple_2l": AQUA}
MODEL_LABEL = {"pile_4l": "Pile 4-layer", "simple_2l": "SimpleStories 2-layer"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "figure.dpi": 150,
    "legend.frameon": False,
})


def load_results(name):
    z = np.load(HERE / "results" / f"{name}.npz")
    return z["loss"], z["dist"].astype(np.int32), z["target_is_eos"]


def log_bins(lo, hi, n=28):
    """Integer-aligned log-spaced bin edges in [lo, hi]."""
    return np.unique(np.geomspace(lo, hi, n).round().astype(int))


def binned_mean(x, y, edges):
    """Mean and SEM of y in each x-bin; returns (geometric bin centers, mean, sem)."""
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    idx, y = idx[ok], y[ok]
    n = np.bincount(idx, minlength=len(edges) - 1)
    mean = np.bincount(idx, y, minlength=len(edges) - 1) / np.maximum(n, 1)
    var = np.bincount(idx, (y - mean[idx]) ** 2, minlength=len(edges) - 1) / np.maximum(n - 1, 1)
    sem = np.sqrt(var / np.maximum(n, 1))
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = n > 1
    return centers[keep], mean[keep], sem[keep]


def power_law(n, L_inf, A, alpha):
    return L_inf + A * n ** -alpha


def fit_power_law(n, y, sem):
    (L_inf, A, alpha), pcov = curve_fit(
        power_law, n, y, p0=[y.min(), y[0] - y.min(), 0.5], sigma=sem, maxfev=10000
    )
    return (L_inf, A, alpha), np.sqrt(np.diag(pcov))


def unigram_entropy(name):
    """Entropy (nats) of the empirical token distribution of the eval sample."""
    if name == "pile_4l":
        rows = torch.stack(torch.load(HERE / "cache" / "pile_rows.pt", weights_only=True))
        ids = rows.ravel().numpy()
    else:
        stories = torch.load(HERE / "cache" / "simple_stories.pt", weights_only=True)
        ids = torch.cat(stories).numpy()
    p = np.bincount(ids) / len(ids)
    p = p[p > 0]
    return -(p * np.log(p)).sum()


stats_lines = []


def note(s):
    print(s)
    stats_lines.append(s)


# ---------- Fig 1: loss vs absolute context length (as trained) ----------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name in MODELS:
    loss, dist, _ = load_results(name)
    n_ctx = np.arange(1, loss.shape[1] + 1)  # column c is a prediction from c+1 context tokens
    mean, sem = loss.mean(0), loss.std(0) / np.sqrt(loss.shape[0])
    (L_inf, A, alpha), err = fit_power_law(n_ctx, mean, sem)
    note(f"{name}: overall mean loss {loss.mean():.4f} nats ({loss.mean()/np.log(2):.3f} bits) | "
         f"fit vs abs position: L_inf={L_inf:.3f}+-{err[0]:.3f}, A={A:.3f}+-{err[1]:.3f}, "
         f"alpha={alpha:.3f}+-{err[2]:.3f}")
    c = MODEL_COLOR[name]
    cx, cm, cs = binned_mean(n_ctx.repeat(loss.shape[0]), loss.T.ravel(), log_bins(1, 513, 40))
    ax.plot(cx, cm, color=c, lw=2, label=MODEL_LABEL[name], zorder=3)
    ax.fill_between(cx, cm - cs, cm + cs, color=c, alpha=0.25, lw=0)
    ax.plot(n_ctx, power_law(n_ctx, L_inf, A, alpha), color=INK, lw=1, ls=":", zorder=2)
    ax.annotate(f"$L_\\infty$={L_inf:.2f}, $\\alpha$={alpha:.2f}", xy=(cx[-1], cm[-1]),
                xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=9, color=c)
ax.set_xscale("log")
ax.set_xlabel("context length  $n$  (tokens, absolute position in row)")
ax.set_ylabel("cross-entropy loss (nats)")
ax.set_title("Loss vs context length — as trained (packed rows)")
ax.plot([], [], color=INK, lw=1, ls=":", label=r"fit  $L_\infty + A\,n^{-\alpha}$")
ax.legend(loc="upper right")
ax.grid(axis="x", which="both", visible=False)
fig.savefig(HERE.parent / "loss_vs_position.png", bbox_inches="tight")


# ---------- Fig 2: truncated context vs document position ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, name in zip(axes, MODELS):
    loss, dist, _ = load_results(name)
    pos1 = np.broadcast_to(np.arange(1, loss.shape[1] + 1), loss.shape)  # t+1

    cx, cm, cs = binned_mean(pos1.ravel(), loss.ravel(), log_bins(1, 513, 30))
    ax.plot(cx, cm, color=MUTED, lw=1.5, label="all tokens, vs position (mixture)")

    trunc = dist == -1  # mid-doc token, doc start cut off: n = t+1 same-doc tokens
    cx, cm, cs = binned_mean(pos1[trunc], loss[trunc], log_bins(1, 513, 30))
    ax.plot(cx, cm, color=BLUE, lw=2, label="mid-doc token, $n$ visible doc tokens\n(start truncated)")
    ax.fill_between(cx, cm - cs, cm + cs, color=BLUE, alpha=0.25, lw=0)
    (L_inf, A, alpha), err = fit_power_law(cx, cm, cs)
    note(f"{name}: fit truncated-context curve: L_inf={L_inf:.3f}+-{err[0]:.3f}, "
         f"A={A:.3f}+-{err[1]:.3f}, alpha={alpha:.3f}+-{err[2]:.3f}")

    fresh = dist >= 1  # doc started in-row: whole doc visible, d = position in doc
    cx, cm, cs = binned_mean(dist[fresh].astype(float), loss[fresh], log_bins(1, 513, 30))
    ax.plot(cx, cm, color=AQUA, lw=2, label="token at doc position $d$\n(whole doc visible)")
    ax.fill_between(cx, cm - cs, cm + cs, color=AQUA, alpha=0.25, lw=0)

    d0 = loss[dist == 0]
    note(f"{name}: first-token-of-doc loss (d=0): {d0.mean():.3f} nats (n={d0.size}) | "
         f"unigram entropy of corpus sample: {unigram_entropy(name):.3f} nats")

    ax.set_xscale("log")
    ax.set_title(MODEL_LABEL[name])
    ax.set_xlabel("same-document context / document position (tokens)")
    ax.grid(axis="x", which="both", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
axes[0].set_ylabel("cross-entropy loss (nats)")
fig.suptitle("Context truncation vs position in document", y=1.02)
fig.savefig(HERE.parent / "truncation_vs_position.png", bbox_inches="tight")


# ---------- Fig 3: does out-of-document context help? ----------
O_BANDS = [(0, 32), (32, 128), (128, 512)]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, name in zip(axes, MODELS):
    loss, dist, _ = load_results(name)
    pos = np.broadcast_to(np.arange(loss.shape[1]), loss.shape)  # t = col index
    o = pos - dist                                               # out-of-doc prefix tokens
    fresh = dist >= 1
    for (lo, hi), c in zip(O_BANDS, ORDINAL_BLUES):
        sel = fresh & (o >= lo) & (o < hi)
        if sel.sum() < 100:
            continue
        cx, cm, cs = binned_mean(dist[sel].astype(float), loss[sel], log_bins(1, 513, 20))
        ax.plot(cx, cm, color=c, lw=2, label=f"$o \\in [{lo}, {hi})$")
        ax.fill_between(cx, cm - cs, cm + cs, color=c, alpha=0.25, lw=0)
    ax.set_xscale("log")
    ax.set_title(MODEL_LABEL[name])
    ax.set_xlabel("document position $d$ (tokens)")
    ax.grid(axis="x", which="both", visible=False)
    ax.legend(loc="upper right", fontsize=8.5, title="out-of-doc prefix $o$", title_fontsize=8.5)
axes[0].set_ylabel("cross-entropy loss (nats)")
fig.suptitle("Same document position, different amounts of unrelated prefix", y=1.02)
fig.savefig(HERE.parent / "out_of_doc_context.png", bbox_inches="tight")

(HERE.parent / "fit_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
print("saved figures + fit_stats.txt")
