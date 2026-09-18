"""Per-component read-in analysis for the top-20 mean-CI components of one
matrix of decomposition C (supersedes the single-component v_dot_embeddings.py /
v_dot_vs_ci.py, same figures). Matrix from argv (default h.0.attn.q_proj):
    python analyze_top20_C.py h.0.attn.k_proj

For each component c (files ../v_dot_embeddings/<matrix>/<mean CI:.2f>-CI-<id>.png
and ../v_dot_vs_ci/<matrix>/<same>.png):
  v_dot_embeddings.png — histograms of a_t = V_c . rms_1(wte[t]) (the actual L0
    input activation, g * wte[t]/rms; majority-positive gauge), equal-weight and
    corpus-frequency-weighted; missing tokens excluded. As many token labels as
    fit without overlapping text (labeling.py; isolated tokens always labeled).
  v_dot_vs_ci.png — scatter of a_t vs the component's mean CI on token t over the
    4,000 cached Pile rows (tokens with count > 0), colored by log10 count.

Requires cache/ci_per_token_<short>_top20.npz from ci_per_token_modal.py
(short = h0q, h0k, ...).

⚠ CI values via the public JAX sink loader = broken RoPE (rope_report.md);
token-identity-level statistics expected to survive qualitatively.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

MOD = sys.argv[1] if len(sys.argv) > 1 else "h.0.attn.q_proj"
BLUE, INK = "#3D6DE2", "#39485E"


def short_name(mod: str) -> str:  # same rule as ci_per_token_modal.py
    p = mod.split(".")
    return f"h{p[1]}{p[3][0] if p[2] == 'attn' else p[3][0] + 'm'}"


sys.path.insert(0, str(HERE))
from labeling import label_order, add_hist_labels, add_scatter_labels

# --- shared data ---
uv = np.load(ROOT / "compare-decomps/hide/cache/uv_C.npz")
V = uv[f"{MOD}|V"].astype(np.float32)  # (C, d_in): rows are components!
s = np.load(ROOT / "compare-decomps/hide/cache/act_signs_C.npz")
mean_ci_comp = np.load(ROOT / "compare-decomps/hide/cache/coci_C.npz")[f"{MOD}|mean"]

with safe_open(ROOT / "sink-models/pretrain_cache/spd-t-87f91319/model_step_100000.safetensors",
               framework="np") as f:
    wte = f.get_tensor("wte.weight").astype(np.float32)
    g = f.get_tensor("h.0.rms_1.weight").astype(np.float32)
rms = np.linalg.norm(wte, axis=1) / np.sqrt(wte.shape[1])
X = (wte / rms[:, None]) * g  # actual L0 attn input per token (rms_1)

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
counts = torch.bincount(torch.cat(list(rows)), minlength=len(wte)).numpy()
missing = np.load(ROOT / "token-embeds/hide/missing_tokens.npy")
keep_hist = np.ones(len(wte), dtype=bool)
keep_hist[missing] = False
keep_ci = keep_hist & (counts > 0)

z = np.load(HERE / "cache" / f"ci_per_token_{short_name(MOD)}_top20.npz")
ids, ci_sum = z["ids"], z["ci_sum"]  # (20,), (vocab, 20)

tok = load_tokenizer("pile_4l")
tt = np.load(ROOT / "mean-ci-widget/hide/cache/top_tokens_C.npz")


def comp_sign(c: int) -> float:
    Npos, F = int(s[f"{MOD}|Npos"][c]), int(s[f"{MOD}|F"][c])
    Ssum, Sall = float(s[f"{MOD}|Ssum"][c]), float(s[f"{MOD}|Sall"][c])
    if F > 0:
        return 1.0 if (Npos * 2 > F or (Npos * 2 == F and Ssum >= 0)) else -1.0
    return 1.0 if Sall >= 0 else -1.0


def dec(i: int) -> str:
    t = tok.decode([int(i)])
    return t.strip() or repr(t)


def dec_plot(i: int) -> str:
    return dec(i).replace("$", r"\$")  # keep mathtext parser away from '$' tokens


kept_ids = np.nonzero(keep_hist)[0]
kept_names = [dec_plot(i) for i in kept_ids]  # decoded once, reused per figure
kept_counts = counts[keep_hist]
ci_names = [n for n, i in zip(kept_names, kept_ids) if counts[i] > 0]  # keep_ci subset
ci_counts = counts[keep_ci]

for k, c in enumerate(ids):
    c = int(c)
    dots = X @ (comp_sign(c) * V[c])
    mean_ci = np.zeros(len(wte))
    mean_ci[keep_ci] = ci_sum[keep_ci, k] / counts[keep_ci]
    tag = f"{mean_ci_comp[c]:.2f}-CI-{c}"
    emb_dir = HERE.parent / "C" / "v_dot_embeddings" / MOD
    sca_dir = HERE.parent / "C" / "v_dot_vs_ci" / MOD
    emb_dir.mkdir(parents=True, exist_ok=True)
    sca_dir.mkdir(parents=True, exist_ok=True)

    top_ids, top_ci = tt[f"{MOD}|top_ids"][c], tt[f"{MOD}|top_ci"][c]
    tot = float(tt[f"{MOD}|total"][c])
    top_str = ", ".join(f"{tok.decode([int(i)])!r} {sc / tot:.0%}"
                        for i, sc in zip(top_ids[:5], top_ci[:5]) if sc > 0)
    print(f"\n=== {MOD}:{c}  mean CI {mean_ci_comp[c]:.4f}  sign {comp_sign(c):+.0f}"
          f"\n    top firing: {top_str}")
    d = dots[keep_hist]
    order = np.argsort(dots)
    neg = ", ".join(f"{dec(i)} {dots[i]:+.1f}" for i in order[:6])
    pos = ", ".join(f"{dec(i)} {dots[i]:+.1f}" for i in order[::-1][:6])
    print(f"    most negative: {neg}\n    most positive: {pos}")

    # --- histogram figure ---
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, w, title, ylab in [
        (axes[0], None, "each token weighted equally", "tokens per bin"),
        (axes[1], counts[keep_hist], "weighted by corpus frequency", "corpus tokens per bin"),
    ]:
        ax.hist(d, bins=200, weights=w, color=BLUE, edgecolor="none")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel(r"$V_c \cdot \mathrm{rms}_1(\mathrm{wte}[t])$")
    fig.suptitle(f"C  {MOD}:{c} — read-in $V_c$ · RMSNorm-ed token embeddings "
                 r"($x_t = g \odot \mathrm{wte}[t]/\mathrm{rms}$; missing tokens excluded)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.canvas.draw()  # finalize transforms for label placement
    lorder = label_order(d, kept_counts)
    n0 = add_hist_labels(axes[0], fig, d, kept_names, lorder)
    # frequency-weighted panel: count-0 tokens have no bar there — skip them
    n1 = add_hist_labels(axes[1], fig, d, kept_names,
                         lorder[kept_counts[lorder] > 0], weights=kept_counts)
    print(f"    labels: {n0} equal-weight panel, {n1} frequency panel")
    fig.savefig(emb_dir / f"{tag}.png", dpi=150)
    plt.close(fig)

    # --- scatter figure ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    o = np.argsort(counts[keep_ci])  # draw high-count points on top
    sc = ax.scatter(dots[keep_ci][o], mean_ci[keep_ci][o],
                    c=np.log10(counts[keep_ci][o]), cmap="viridis",
                    s=8, alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$ corpus count")
    ax.set_xlabel(r"$V_c \cdot \mathrm{rms}_1(\mathrm{wte}[t])$")
    ax.set_ylabel("mean CI on token $t$")
    ax.set_title(f"C  {MOD}:{c} — input activation vs per-token mean CI\n"
                 r"($x_t = g \odot \mathrm{wte}[t]/\mathrm{rms}$; 4,000 Pile rows; "
                 "missing tokens excluded)", fontsize=11, color=INK)
    ax.grid(color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()  # finalize transforms for label placement
    ns = add_scatter_labels(ax, fig, dots[keep_ci], mean_ci[keep_ci],
                            ci_names, ci_counts)
    print(f"    scatter labels: {ns}")
    fig.savefig(sca_dir / f"{tag}.png", dpi=150)
    plt.close(fig)
    print(f"    saved -> C/v_dot_embeddings/{MOD}/{tag}.png + C/v_dot_vs_ci/{MOD}/{tag}.png")
