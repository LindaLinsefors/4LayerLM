"""Which L<l> q/k components have position-in-chunk-dependent causal importance?

From cache/pos_ci.npz (per-position CI sums Sp and CI>0.1 fire counts Fp over
4000 Pile rows, all 8 q/k matrices; see pos_ci_compute_modal.py). Per alive
component:

  profile   w_b  = fires in position bin b / total fires (64 bins x 8 positions)
  TV        0.5 * sum_b |w_b - 1/64|      (0 = flat, 1 = fully displaced)
  noise TV  Monte-Carlo mean TV of n_fires uniform draws (raw TV is inflated
            for small n), and excess TV = TV - noise TV  (the ranking metric)

The same is computed for activation firings (|a_c| > 1); for L0 the module
input is a function of the token id only, so those are a null control.

Besides the fire-count heatmaps, mean heatmaps show the per-position mean of
CI (Sp/4000) and |a_c| (Ap/4000), each divided by the component's overall
mean so rows are comparable — same ratio-to-flat reading as the count plots.

Writes L<l>/L<l>_position_dependence.md + heatmaps in hide/figures/L<l>-pos-ci/.

Usage: python pos_ci_report.py [layer ...]   (default: 0)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = Path(__file__).parent
OUT_BASE = HERE.parent

POS = np.load(HERE / "cache" / "pos_ci.npz")
MEAN_CI = np.load(HERE / "cache" / "mean_ci.npz")
SITE = json.load(open(HERE / "cache" / "site_descriptions.json"))

N_BINS, HEAVY, MODERATE = 64, 0.5, 0.25
rng = np.random.default_rng(0)


def noise_tv(n: int, sims: int = 200) -> float:
    """Expected TV of n uniform firings over N_BINS bins (sampling noise)."""
    if n == 0:
        return 0.0
    counts = rng.multinomial(n, np.full(N_BINS, 1 / N_BINS), size=sims)
    return float(0.5 * np.abs(counts / n - 1 / N_BINS).sum(1).mean())


def profile_tv(f: np.ndarray) -> tuple[int, float]:
    n = int(f.sum())
    if n == 0:
        return 0, 0.0
    w = f.reshape(N_BINS, -1).sum(1) / n
    return n, 0.5 * np.abs(w - 1 / N_BINS).sum()


def analyze(mod: str) -> list[dict]:
    idx = POS[f"{mod}|idx"]
    Fp = POS[f"{mod}|Fp"]          # CI fires,        (C_alive, 512)
    Fa = POS[f"{mod}|Fa"]          # activation fires, (C_alive, 512)
    Sp = POS[f"{mod}|Sp"]          # CI sum,           (C_alive, 512)
    Ap = POS[f"{mod}|Ap"]          # |a_c| sum,        (C_alive, 512)
    site_key = mod.replace("h.", "").replace("_proj", "")
    rows = []
    for k, c in enumerate(idx):
        f, fa = Fp[k], Fa[k]
        n, tv = profile_tv(f)
        na, tva = profile_tv(fa)
        cum = np.cumsum(f)
        rows.append({
            "comp": int(c), "n": n, "tv": tv, "noise": noise_tv(n),
            "na": na, "tva": tva, "noise_a": noise_tv(na),
            "median": int(np.searchsorted(cum, n / 2)) if n else -1,
            "f8": f[:8].sum() / n if n else 0.0,
            "f32": f[:32].sum() / n if n else 0.0,
            "label": SITE[f"{site_key}:{c}"]["label"],
            "mci": float(MEAN_CI[mod][c]),
            "profile": f, "a_profile": fa,
            "ci_mean": Sp[k], "a_mean": Ap[k],
        })
    for r in rows:
        r["excess"] = r["tv"] - r["noise"]
        r["excess_a"] = r["tva"] - r["noise_a"]
    rows.sort(key=lambda r: -r["excess"])
    return rows


def heatmap(mod: str, rows: list[dict], fig_path: Path, key: str, what: str,
            exc: str, cbar: str, ylab_note: str) -> None:
    n_comp = len(rows)
    prof = np.stack([r[key] for r in rows])
    tot = prof.sum(1, keepdims=True).clip(min=1e-30)
    coarse = prof.reshape(n_comp, 128, 4).sum(2) / tot / (4 / 512)   # ratio to uniform
    early = prof[:, :16] / tot / (1 / 512)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 0.11 * n_comp + 2), constrained_layout=True,
        gridspec_kw={"width_ratios": [128, 24]})
    pos = np.concatenate([coarse.ravel(), early.ravel()])
    pos = pos[pos > 0]
    vmin, vmax = pos.min(), pos.max()
    if vmax / vmin < 1.5:                      # keep a readable scale when ~flat
        vmin, vmax = vmin / 1.25, vmax * 1.25
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap("viridis").copy()
    cm.set_under("#f0f0f0")
    for ax, M, x_edges, xlab in (
            (axes[0], coarse, np.arange(0, 513, 4), "chunk position (bins of 4)"),
            (axes[1], early, np.arange(17), "position (first 16)")):
        im = ax.pcolormesh(x_edges, np.arange(n_comp + 1), M, norm=norm, cmap=cm)
        ax.invert_yaxis()
        ax.set_xlabel(xlab, fontsize=9)
        labels = [f"{r['comp']}  ({r[exc]:.2f})" for r in rows]
        ax.set_yticks(np.arange(n_comp) + 0.5,
                      labels if ax is axes[0] else [], fontsize=4.5)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.5, label=f"{cbar} (log, per-figure scale)")
    fig.suptitle(f"{mod} — {what} position profiles, all alive components\n"
                 f"rows sorted by CI excess TV (y label: comp id + "
                 f"{ylab_note}); gray = exactly zero in bin", fontsize=11)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def table(rows: list[dict], cut: float) -> list[str]:
    out = ["| comp | fires | TV | excess TV | median pos | % pos<8 | % pos<32 "
           "| a-fires | a-excess TV | label |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["excess"] < cut:
            continue
        mark = "**" if r["excess"] >= HEAVY else ""
        out.append(
            f"| {mark}{r['comp']}{mark} | {r['n']} | {r['tv']:.2f} | "
            f"{r['excess']:.2f} | {r['median']} | {100 * r['f8']:.0f} | "
            f"{100 * r['f32']:.0f} | {r['na']} | {r['excess_a']:.2f} | "
            f"{r['label']} |")
    return out


def main(layer: int) -> None:
    fig_dir = HERE / "figures" / f"L{layer}-pos-ci"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# L{layer} — q/k components with position-in-chunk-dependent "
        "causal importance",
        "",
        f"Per alive component of `h.{layer}.attn.q_proj` / "
        f"`h.{layer}.attn.k_proj`: where in "
        "the 512-token chunk do its CI firings (CI > 0.1) happen? Over the "
        "4,000 cached Pile rows every chunk position is seen exactly 4,000 "
        "times, and chunk boundaries fall at random points inside documents, "
        "so the token distribution is the same at every position (position 0 "
        "is *not* usually a document start — 74.6% of chunks contain no EOS). "
        "A component whose CI depends only on token/context therefore has a "
        "flat firing-position profile; any structure is genuine position "
        "dependence. Computed on Modal (`hide/pos_ci_compute_modal.py` → "
        "`hide/cache/pos_ci.npz`); tables/figures from `hide/pos_ci_report.py`.",
        "",
        "**Metric:** profile $w_b$ = share of fires in position bin $b$ "
        "(64 bins × 8 positions); $TV = \\tfrac12\\sum_b |w_b - 1/64|$ "
        "(0 = position-independent, 1 = fully displaced from uniform). Raw TV "
        "is inflated for components with few fires, so we subtract the "
        "Monte-Carlo expectation for the same number of uniform fires: "
        "**excess TV** (≈ 0 for flat components). Components with excess TV "
        f"≥ {HEAVY} are **heavily** position-dependent (bold); the tables "
        f"list everything ≥ {MODERATE}.",
        "",
        "**Activation firings (control):** the same profile/metric for "
        "*activation* firings, $|a_c| > 1$ with "
        "$a_c = \\|U_c\\|\\,(V_c\\cdot\\varphi)$ (the norm of the component's "
        "rank-one write into the q/k output) — columns `a-fires`, "
        "`a-excess TV` and their own heatmaps. For **L0** the module input "
        "$\\varphi$ is a function of the token id alone (post-embedding "
        "RMSNorm; position enters attention only via RoPE, downstream of "
        "q/k-space), so L0 activation profiles are position-independent by "
        "architecture — a null control for the pipeline. For L1 the input "
        "carries whatever layer 0 wrote, so positional activations are "
        "possible.",
        "",
        "**Position 0 is special** (visible in the median-pos column): q "
        "components never have CI there — with only one visible key, "
        "softmax over a single logit is constant, so the query has zero "
        "causal effect at position 0 (holds for all layers: ≤ 37 stray "
        "q fires at pos 0 model-wide vs thousands at pos 1). k components "
        "show the mirror image: key 0 is read by every later query (the "
        "attention-sink site), so k CI is *largest* at position 0.",
        "",
        "**Heatmaps:** four per matrix. *Firing* heatmaps show fire-count "
        "density relative to uniform (a value of 10 at a position = 10× more "
        "of the component's fires land there than a flat profile would put "
        "there). *Mean* heatmaps show the per-position mean of CI "
        "($S_p/4000$) resp. $|a_c|$ ($A_p/4000$), divided by the component's "
        "overall mean — the same ratio-to-flat reading, but weighted by "
        "magnitude instead of binarized. Rows are sorted by CI excess TV in "
        "all four. Each figure's color scale spans its own data range (the "
        "scales are **not** comparable across figures).",
        "",
    ]
    chat_summary = {}
    for mod, short in ((f"h.{layer}.attn.q_proj", "q"),
                       (f"h.{layer}.attn.k_proj", "k")):
        rows = analyze(mod)
        flat = [r["excess"] for r in rows if r["excess"] < MODERATE]
        a_dep = [r for r in rows if r["excess_a"] >= MODERATE]
        a_max = max(r["excess_a"] for r in rows)
        a_note = (
            "Activation firings: " + (
                f"max a-excess TV = {a_max:.2f} — no component's *activation* "
                "is position-dependent."
                if not a_dep else
                f"{len(a_dep)} components have a-excess TV ≥ {MODERATE}: "
                + ", ".join(f"{r['comp']} ({r['excess_a']:.2f})" for r in a_dep)
                + "."))
        lines += [
            f"## {mod}",
            "",
            f"{len(rows)} alive components; "
            f"{sum(r['excess'] >= HEAVY for r in rows)} heavy, "
            f"{sum(HEAVY > r['excess'] >= MODERATE for r in rows)} moderate. "
            f"The remaining {len(flat)} have excess TV ≤ "
            f"{max(flat):.2f} (median {np.median(flat):.2f}) — "
            "position-independent within noise. " + a_note,
            "",
            *table(rows, MODERATE),
            "",
            f"![{mod} CI firing profiles]"
            f"(../hide/figures/{fig_dir.name}/{short}.png)",
            "",
            f"![{mod} mean CI profiles]"
            f"(../hide/figures/{fig_dir.name}/{short}_mean.png)",
            "",
            f"![{mod} activation firing profiles]"
            f"(../hide/figures/{fig_dir.name}/{short}_act.png)",
            "",
            f"![{mod} mean activation profiles]"
            f"(../hide/figures/{fig_dir.name}/{short}_act_mean.png)",
            "",
        ]
        heatmap(mod, rows, fig_dir / f"{short}.png",
                key="profile", what="CI-firing (CI > 0.1)",
                exc="excess", cbar="fire density / uniform",
                ylab_note="CI excess TV")
        heatmap(mod, rows, fig_dir / f"{short}_mean.png",
                key="ci_mean", what="mean-CI",
                exc="excess", cbar="position mean CI / overall mean",
                ylab_note="CI excess TV")
        heatmap(mod, rows, fig_dir / f"{short}_act.png",
                key="a_profile", what="activation-firing (|a| > 1)",
                exc="excess_a", cbar="fire density / uniform",
                ylab_note="activation excess TV")
        heatmap(mod, rows, fig_dir / f"{short}_act_mean.png",
                key="a_mean", what="mean-|a|",
                exc="excess_a", cbar="position mean |a| / overall mean",
                ylab_note="activation excess TV")
        chat_summary[short] = [(r["comp"], r["excess"]) for r in rows
                               if r["excess"] >= MODERATE]

    layer_dir = OUT_BASE / f"L{layer}"
    layer_dir.mkdir(exist_ok=True)
    report_path = layer_dir / f"L{layer}_position_dependence.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report_path)
    for short, comps in chat_summary.items():
        print(f"{short}: " + ", ".join(f"{c} ({e:.2f})" for c, e in comps))


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["0"]:
        main(int(arg))
