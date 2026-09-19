"""Per-component read-in analysis for the top-20 sample-mean-CI components of
h.0.attn.q_proj and h.0.attn.k_proj of decompositions A, B (pile_4l target
t-9d2b8f02), D, E (sink targets t-87f91319 / t-75f6c439) and F (p-c45e0001,
corrected-RoPE re-decomposition of t-87f91319) — the analyze_top20_{C,old}
analogue. Decompositions from argv (default all five):
    python analyze_top20_new.py A B D E F

Outputs ../<dec>/{v_dot_embeddings,v_dot_vs_ci}/<matrix>/<mean CI:.2f>-CI-<id>.png
(same figure formats + dense labels as components/C, via labeling.py).

V factors: uv caches store V as (C, d_in) — ROW c is component c's read-in
(uv_newA/newB are alive-only; row index via cross_alive.npz). Sign gauge =
majority-positive-activation from the act_signs caches. Requires
cache/ci_per_token_<dec>_<h0q|h0k>_top20.npz from ci_per_token_modal_{AB,DE}.py.

⚠ D/E CI values via the public JAX sink loader = broken RoPE (rope_report.md);
token-identity-level statistics expected to survive qualitatively. A/B are
clean; F is clean too (its caches were computed under the fitted spectrum,
which IS F's training-time forward).
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
sys.path.insert(0, str(HERE))
from load import load_tokenizer
from labeling import label_order, add_hist_labels, add_scatter_labels

PILE_TARGET = "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
DECS = {
    "A": dict(uv="coci-heatmaps/hide/cache/uv_newA.npz", alive_key="newA",
              signs="coci-heatmaps/hide/cache/act_signs_newA.npz",
              coci="coci-heatmaps/hide/cache/coci_newA.npz",
              tt="mean-ci-widget/hide/cache/top_tokens_newA.npz", target=PILE_TARGET),
    "B": dict(uv="coci-heatmaps/hide/cache/uv_newB.npz", alive_key="newB",
              signs="coci-heatmaps/hide/cache/act_signs_newB.npz",
              coci="coci-heatmaps/hide/cache/coci_newB.npz",
              tt="mean-ci-widget/hide/cache/top_tokens_newB.npz", target=PILE_TARGET),
    "D": dict(uv="compare-decomps/hide/cache/uv_D.npz", alive_key=None,
              signs="compare-decomps/hide/cache/act_signs_D.npz",
              coci="compare-decomps/hide/cache/coci_D.npz",
              tt="mean-ci-widget/hide/cache/top_tokens_D.npz",
              target="sink-models/pretrain_cache/spd-t-87f91319/model_step_100000.safetensors"),
    "E": dict(uv="compare-decomps/hide/cache/uv_E.npz", alive_key=None,
              signs="compare-decomps/hide/cache/act_signs_E.npz",
              coci="compare-decomps/hide/cache/coci_E.npz",
              tt="mean-ci-widget/hide/cache/top_tokens_E.npz",
              target="sink-models/pretrain_cache/spd-t-75f6c439/model_step_100000.safetensors"),
    "F": dict(uv="compare-decomps/hide/cache/uv_F.npz", alive_key=None,
              signs="compare-decomps/hide/cache/act_signs_F.npz",
              coci="coci-heatmaps/hide/cache/coci_F.npz",
              tt="mean-ci-widget/hide/cache/top_tokens_F.npz",
              target="sink-models/pretrain_cache/spd-t-87f91319/model_step_100000.safetensors"),
}
MODS = ("h.0.attn.q_proj", "h.0.attn.k_proj")
BLUE, INK = "#3D6DE2", "#39485E"
run_decs = sys.argv[1:] or list(DECS)


def short_name(mod: str) -> str:  # same rule as ci_per_token_modal_*.py
    p = mod.split(".")
    return f"h{p[1]}{p[3][0] if p[2] == 'attn' else p[3][0] + 'm'}"


# --- shared token data (same tokenizer/rows for all decs) ---
tok = load_tokenizer("pile_4l")
rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")
counts = torch.bincount(torch.cat(list(rows)), minlength=50277).numpy()
missing = np.load(ROOT / "token-embeds/hide/missing_tokens.npy")
keep_hist = np.ones(len(counts), dtype=bool)
keep_hist[missing] = False
keep_ci = keep_hist & (counts > 0)


def dec_tok(i: int) -> str:
    t = tok.decode([int(i)])
    return t.strip() or repr(t)


def name_plot(i: int) -> str:
    return dec_tok(i).replace("$", r"\$")  # keep mathtext away from '$' tokens


kept_ids = np.nonzero(keep_hist)[0]
kept_names = [name_plot(i) for i in kept_ids]
kept_counts = counts[keep_hist]
ci_names = [n for n, i in zip(kept_names, kept_ids) if counts[i] > 0]
ci_counts = counts[keep_ci]

cross_alive = np.load(ROOT / "coci-heatmaps/hide/cache/cross_alive.npz")

for DEC in run_decs:
    cfg = DECS[DEC]
    uv = np.load(ROOT / cfg["uv"])
    s = np.load(ROOT / cfg["signs"])
    coci = np.load(ROOT / cfg["coci"])
    tt = np.load(ROOT / cfg["tt"])
    with safe_open(ROOT / cfg["target"], framework="np") as f:
        wte = f.get_tensor("wte.weight").astype(np.float32)
        g = f.get_tensor("h.0.rms_1.weight").astype(np.float32)
    rms = np.linalg.norm(wte, axis=1) / np.sqrt(wte.shape[1])
    X = (wte / rms[:, None]) * g  # actual L0 attn input per token (rms_1)

    def comp_sign(mod: str, c: int) -> float:
        Npos, F = int(s[f"{mod}|Npos"][c]), int(s[f"{mod}|F"][c])
        Ssum, Sall = float(s[f"{mod}|Ssum"][c]), float(s[f"{mod}|Sall"][c])
        if F > 0:
            return 1.0 if (Npos * 2 > F or (Npos * 2 == F and Ssum >= 0)) else -1.0
        return 1.0 if Sall >= 0 else -1.0

    def read_in(mod: str, c: int) -> np.ndarray:
        V = uv[f"{mod}|V"]  # (C or n_alive, d_in) — rows are components
        if cfg["alive_key"] is None:
            return V[c].astype(np.float32)
        alive = cross_alive[f"{cfg['alive_key']}|{mod}"]
        pos = np.nonzero(alive == c)[0]
        assert len(pos) == 1, f"{DEC} {mod}:{c} not in alive set"
        return V[int(pos[0])].astype(np.float32)

    for MOD in MODS:
        mean_ci_comp = coci[f"{MOD}|mean"]
        z = np.load(HERE / "cache" / f"ci_per_token_{DEC}_{short_name(MOD)}_top20.npz")
        ids, ci_sum = z["ids"], z["ci_sum"]  # (20,), (vocab, 20)

        for k, c in enumerate(ids):
            c = int(c)
            dots = X @ (comp_sign(MOD, c) * read_in(MOD, c))
            mean_ci = np.zeros(len(wte))
            mean_ci[keep_ci] = ci_sum[keep_ci, k] / counts[keep_ci]
            tag = f"{mean_ci_comp[c]:.2f}-CI-{c}"
            emb_dir = HERE.parent / DEC / "v_dot_embeddings" / MOD
            sca_dir = HERE.parent / DEC / "v_dot_vs_ci" / MOD
            emb_dir.mkdir(parents=True, exist_ok=True)
            sca_dir.mkdir(parents=True, exist_ok=True)

            top_ids, top_ci = tt[f"{MOD}|top_ids"][c], tt[f"{MOD}|top_ci"][c]
            tot = float(tt[f"{MOD}|total"][c])
            top_str = ", ".join(f"{tok.decode([int(i)])!r} {sc / tot:.0%}"
                                for i, sc in zip(top_ids[:5], top_ci[:5]) if sc > 0)
            print(f"\n=== {DEC} {MOD}:{c}  mean CI {mean_ci_comp[c]:.4f}  "
                  f"sign {comp_sign(MOD, c):+.0f}\n    top firing: {top_str}")
            d = dots[keep_hist]
            order = np.argsort(dots)
            neg = ", ".join(f"{dec_tok(i)} {dots[i]:+.1f}" for i in order[:6])
            pos = ", ".join(f"{dec_tok(i)} {dots[i]:+.1f}" for i in order[::-1][:6])
            print(f"    most negative: {neg}\n    most positive: {pos}")

            # --- histogram figure ---
            fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            for ax, w, title, ylab in [
                (axes[0], None, "each token weighted equally", "tokens per bin"),
                (axes[1], kept_counts, "weighted by corpus frequency",
                 "corpus tokens per bin"),
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
            fig.suptitle(f"{DEC}  {MOD}:{c} — read-in $V_c$ · RMSNorm-ed token embeddings "
                         r"($x_t = g \odot \mathrm{wte}[t]/\mathrm{rms}$; "
                         "missing tokens excluded)", fontsize=12, color=INK)
            fig.tight_layout()
            fig.canvas.draw()
            lorder = label_order(d, kept_counts)
            n0 = add_hist_labels(axes[0], fig, d, kept_names, lorder)
            n1 = add_hist_labels(axes[1], fig, d, kept_names,
                                 lorder[kept_counts[lorder] > 0], weights=kept_counts)
            print(f"    labels: {n0} equal-weight, {n1} frequency panel")
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
            ax.set_title(f"{DEC}  {MOD}:{c} — input activation vs per-token mean CI\n"
                         r"($x_t = g \odot \mathrm{wte}[t]/\mathrm{rms}$; 4,000 Pile rows; "
                         "missing tokens excluded)", fontsize=11, color=INK)
            ax.grid(color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            fig.canvas.draw()
            ns = add_scatter_labels(ax, fig, dots[keep_ci], mean_ci[keep_ci],
                                    ci_names, ci_counts)
            print(f"    scatter labels: {ns}")
            fig.savefig(sca_dir / f"{tag}.png", dpi=150)
            plt.close(fig)
            print(f"    saved -> {DEC}/*/{MOD}/{tag}.png")
