"""Render per-matrix CI co-activation heatmaps + the two .md reports.

For each model and each decomposed matrix, one heatmap of Pearson r of
per-token CI (from hide/cache/coci_<model>.npz, computed on Modal by
coci_compute_modal.py), components sorted by harvest mean CI descending
(hide/cache/mean_ci_<model>.npz; ties by component id). Two versions:
  alive  -> report_alive.md   (harvest mean CI > 1e-6, project convention)
  all    -> report_all.md     (every component; zero-variance rows are gray)
Figures in hide/figures/<model>/<alive|all>/; reports one level up.

Usage: python coci-heatmaps/hide/report.py   (~2 min, mostly rasterizing)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parent / "old"  # reports moved into old/ 2026-09-02 (old vs new A/B decompositions)
CACHE = HERE / "cache"

N_LAYERS = {"pile_4l": 4, "simple_2l": 2}
MATS = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
        "mlp.c_fc", "mlp.down_proj"]
MODELS = ["pile_4l", "simple_2l"]
TOKENS_DESC = {
    "pile_4l": "4,000 cached Pile training rows (2.05M tokens)",
    "simple_2l": "6,000 cached SimpleStories stories (1.72M tokens)",
}


def heatmap(r: np.ndarray, ids: np.ndarray, title: str, path: Path) -> None:
    n = len(ids)
    side = float(np.clip(2 + n / 220, 5, 18))
    fig, ax = plt.subplots(figsize=(side + 1.2, side), constrained_layout=True)
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(r), cmap=cm, vmin=-1, vmax=1,
                   interpolation="nearest")
    if n <= 120:
        fs = max(5, min(9, int((side - 1) * 72 / n * 0.6)))
        ax.set_xticks(range(n), ids, rotation=90, fontsize=fs)
        ax.set_yticks(range(n), ids, fontsize=fs)
        ax.tick_params(length=0)
    else:
        ax.set_xlabel("rank (sorted by harvest mean CI, descending)", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    lines = {v: [
        f"# CI co-activation per matrix — {t} components",
        "",
        "For each model and each decomposed weight matrix: **Pearson r of the "
        "per-token causal importance** (lower_leaky, `sampling=\"continuous\"`) "
        "between every pair of the matrix's subcomponents, i.e. the co-CI "
        "measure of the pile-qk-comps reports, here for all 6 matrix types. "
        "Samples: " + " / ".join(f"{m} over {TOKENS_DESC[m]}" for m in MODELS)
        + " (SimpleStories batches padded; pad positions excluded).",
        "",
        "Axes: components sorted by **harvest-DB mean CI, descending** (ties "
        "by component id); id tick labels only where they fit (n ≤ 120), "
        "otherwise the axis is the rank. Diagonal = 1. **Gray** = component "
        "with zero CI variance in the sample (r undefined). Alive = harvest "
        "mean CI > 1e-6 (project convention: 9,973 / 6,535 components).",
        "",
    ] for v, t in (("alive", "alive"), ("all", "all"))}

    for model in MODELS:
        stats = np.load(CACHE / f"coci_{model}.npz")
        mci = np.load(CACHE / f"mean_ci_{model}.npz")
        for v in lines:
            lines[v] += [f"## {model}", ""]
        for l in range(N_LAYERS[model]):
            for v in lines:
                lines[v] += [f"### Layer {l}", ""]
            for mat in MATS:
                mod = f"h.{l}.{mat}"
                r = stats[f"{mod}|r"].astype(np.float32)
                m = mci[mod]
                assert len(m) == len(r)
                order_all = np.argsort(-m, kind="stable")
                alive = order_all[m[order_all] > 1e-6]
                n_dead_var = int(np.isnan(np.diag(r)).sum())
                for v, order in (("alive", alive), ("all", order_all)):
                    fig_dir = HERE / "figures" / model / v
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"h{l}_{mat.replace('.', '_')}.png"
                    n = len(order)
                    title = (f"{model}  {mod}\nco-CI r(CI), {n} "
                             + ("alive " if v == "alive" else "")
                             + "components (mean-CI-sorted)")
                    heatmap(r[np.ix_(order, order)], order, title,
                            fig_dir / fname)
                    cap = (f"#### {mod} — {n} alive of {len(m)}" if v == "alive"
                           else f"#### {mod} — all {len(m)} components "
                                f"({len(alive)} alive, {n_dead_var} zero-variance)")
                    lines[v] += [cap, "",
                                 f"![{mod}](../hide/figures/{model}/{v}/{fname})",
                                 ""]
                print(f"{model} {mod}: C={len(m)}, alive={len(alive)}, "
                      f"zero-var={n_dead_var}")

    for v, name in (("alive", "report_alive.md"), ("all", "report_all.md")):
        (OUT / name).write_text("\n".join(lines[v]), encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
