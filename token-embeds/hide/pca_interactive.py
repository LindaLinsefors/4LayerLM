"""Interactive (Plotly) token-embedding PCA, one self-contained HTML per model.

PC directions come from PCA of the ALIVE tokens only (mean-centered on the
alive mean). Each file has two zoomable panels:

  1. Variance along each PC (log y) -- three lines: variance of the alive /
     dead-but-seen / never-seen groups' projections onto the alive-PC
     directions (each group centered on its own per-direction mean; the alive
     line is the PCA eigenvalue spectrum).
  2. Group-mean projection of the raw embeddings onto each PC, sign fixed so
     the alive mean is >= 0 (relative signs between groups preserved).

Outputs: token-embeds/pca_interactive_{pile_4l,simple_2l}.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotly.subplots import make_subplots
from safetensors.torch import load_file

ROOT = Path(__file__).parent.parent.parent

pile_emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
simple_emb = torch.load(
    ROOT / "prev_paper/models/simplestories_2layer/target_model_gf6rbga0/model_step_99999.pt",
    map_location="cpu", weights_only=True,
)["wte.weight"].float().numpy()

pile_rows = torch.stack(torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
))
pile_counts = np.bincount(pile_rows.flatten().numpy(), minlength=len(pile_emb))

table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
simple_counts = np.zeros(len(simple_emb), dtype=np.int64)
simple_counts[table["id"].to_numpy()] = table["freq"].to_numpy()

GROUP_COLORS = {"alive": "#4e79a7", "dead, seen": "#e0793d", "never seen": "#a1443a",
                "frequent": "#4e79a7", "unfrequent": "#e0793d", "missing": "#a1443a"}

missing_ids = np.load(Path(__file__).parent / "missing_tokens.npy")

for name, emb, counts, thr in [
    ("pile_4l", pile_emb, pile_counts, 7),
    ("simple_2l", simple_emb, simple_counts, 10),
]:
    alive = counts >= thr  # pile terminology: "frequent"
    x = emb[alive] - emb[alive].mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    pcs = np.arange(1, len(s) + 1)

    proj = emb @ vt.T
    alive_mean = proj[alive].mean(axis=0)
    proj *= np.where(alive_mean >= 0, 1.0, -1.0)  # sign: alive/frequent mean >= 0

    if name == "pile_4l":
        missing = np.zeros(len(emb), dtype=bool)
        missing[missing_ids] = True
        groups = [("frequent", alive), ("unfrequent", ~alive & ~missing),
                  ("missing", missing)]
    else:
        groups = [("alive", alive), ("dead, seen", (counts > 0) & ~alive),
                  ("never seen", counts == 0)]

    base = "frequent" if name == "pile_4l" else "alive"
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"variance along each {base}-PC direction, per token group",
            f"group-mean projection onto each PC (sign: {base} mean ≥ 0)",
        ),
        vertical_spacing=0.12,
    )
    for label, sel in groups:
        legend_name = f"{label} (n={sel.sum():,})"
        fig.add_scatter(
            x=pcs, y=proj[sel].var(axis=0), mode="lines", row=1, col=1,
            name=legend_name, legendgroup=label,
            line=dict(color=GROUP_COLORS[label], width=1.5),
            hovertemplate="PC %{x}<br>variance %{y:.3e}<extra>" + label + "</extra>",
        )
        fig.add_scatter(
            x=pcs, y=proj[sel].mean(axis=0), mode="lines", row=2, col=1,
            name=legend_name, legendgroup=label, showlegend=False,
            line=dict(color=GROUP_COLORS[label], width=1.5),
            hovertemplate="PC %{x}<br>mean proj %{y:.4f}<extra>" + label + "</extra>",
        )

    fig.update_yaxes(type="log", title_text="variance along PC", row=1, col=1)
    fig.update_yaxes(title_text="group-mean projection", row=2, col=1)
    fig.update_xaxes(title_text="principal component index", row=2, col=1)
    fig.update_layout(
        height=850, template="plotly_white",
        title=f"{name} — token-embedding PCA on {base} tokens "
              f"({alive.sum():,} {base}, d={emb.shape[1]}; "
              "drag to zoom, double-click to reset; legend toggles both panels)",
        legend=dict(groupclick="togglegroup"),
    )

    out = Path(__file__).parent.parent / f"pca_interactive_{name}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
