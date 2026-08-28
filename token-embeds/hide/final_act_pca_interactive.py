"""Interactive (Plotly) PCA of the final-residual-stream activations (after
ln_f, just before the unembedding), one self-contained HTML per model. Analog
of pca_interactive.py for the embeddings, without the token-group splits.

Each file has two zoomable panels:
  1. Variance along each PC (log y) — the eigenvalue spectrum of the
     mean-centered activations.
  2. Mean projection of the raw activations onto each PC (= mu . v_i, sign
     fixed so the mean is >= 0) — which PCs carry the shared offset.

Outputs: token-embeds/final_act_pca_interactive_{pile_4l,simple_2l}.html
"""

from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots

HERE = Path(__file__).parent

MODEL_COLORS = {"pile_4l": "#4e79a7", "simple_2l": "#e0793d"}

for name in ["pile_4l", "simple_2l"]:
    acts = np.load(HERE / "cache" / f"final_acts_{name}.npz")["acts"].astype(np.float64)
    n, d = acts.shape
    mu = acts.mean(axis=0)
    x = acts - mu
    cov = (x.T @ x) / n
    var, v = np.linalg.eigh(cov)
    var, v = var[::-1], v[:, ::-1]
    mean_proj = np.abs(mu @ v)  # sign convention: mean >= 0
    pcs = np.arange(1, d + 1)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "variance along each PC (eigenvalue spectrum)",
            "mean projection of raw activations onto each PC (= μ·v, sign: mean ≥ 0)",
        ),
        vertical_spacing=0.12,
    )
    fig.add_scatter(
        x=pcs, y=var, mode="lines", row=1, col=1, showlegend=False,
        line=dict(color=MODEL_COLORS[name], width=1.5),
        hovertemplate="PC %{x}<br>variance %{y:.3e}<extra></extra>",
    )
    fig.add_scatter(
        x=pcs, y=mean_proj, mode="lines", row=2, col=1, showlegend=False,
        line=dict(color=MODEL_COLORS[name], width=1.5),
        hovertemplate="PC %{x}<br>mean proj %{y:.4f}<extra></extra>",
    )

    fig.update_yaxes(type="log", title_text="variance along PC", row=1, col=1)
    fig.update_yaxes(title_text="mean projection", row=2, col=1)
    fig.update_xaxes(title_text="principal component index", row=2, col=1)
    fig.update_layout(
        height=850, template="plotly_white",
        title=f"{name} — PCA of final-residual-stream (post-ln_f) activations "
              f"({n:,} training-data positions, d={d}; "
              "drag to zoom, double-click to reset)",
    )

    out = HERE.parent / f"final_act_pca_interactive_{name}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
