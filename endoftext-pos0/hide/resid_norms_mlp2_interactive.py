"""Interactive (Plotly) version of the "after MLP 2" panel of
resid_stage_norms.png: histograms of the residual-stream norm after block 2
(h.1 output), split by the same four position categories, but with raw counts
(density=False) and zoomable axes. A linear/log y toggle makes the tiny groups
visible next to the ~511k-position bulk without zooming.

Norms are cached to hide/cache/mlp2_norms.npz on first run (same data as
resid_stage_norms.py: first N_ROWS cached Pile rows, truncated to 512), so
re-rendering the figure needs no forward pass.

Output: endoftext-pos0/resid_norms_mlp2_interactive.html
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

N_ROWS = 1000
BATCH_SIZE = 16
EOS_ID = 0
N_BINS = 300  # finer than the png's 80 — zooming should reveal structure

CACHE = HERE / "cache" / "mlp2_norms.npz"
if CACHE.exists():
    d = np.load(CACHE)
    norms, token_ids = d["norms"], d["token_ids"]
else:
    import torch
    from load import load_pile_4l

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)[:N_ROWS]
    rows = [r[:512] for r in rows]  # rows are 513 tokens, n_ctx = 512

    model, _, _ = load_pile_4l()
    model = model.to(device)
    cap = {}
    hook = model.h[1].register_forward_hook(  # block 1 output = after MLP 2
        lambda _m, _i, out: cap.__setitem__("h", out))
    norms_l, ids_l = [], []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = torch.stack(rows[start : start + BATCH_SIZE]).to(device)
            model(batch)
            norms_l.append(cap["h"].float().norm(dim=-1).flatten().cpu())
            ids_l.append(batch.cpu().flatten())
    hook.remove()
    norms = torch.cat(norms_l).numpy()
    token_ids = torch.cat(ids_l).numpy()
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, norms=norms, token_ids=token_ids)
    print(f"cached {CACHE}")

pos = np.tile(np.arange(512), N_ROWS)
is_eos = token_ids == EOS_ID
groups = [  # (mask, label, color) — draw order: bulk first, small groups on top
    (~is_eos & (pos > 0), "other positions", "#1f77b4"),
    (is_eos & (pos > 0), "<|endoftext|> (mid-sequence)", "#d62728"),
    (~is_eos & (pos == 0), "position 0", "#ff7f0e"),
    (is_eos & (pos == 0), "<|endoftext|> at position 0", "#9467bd"),
]
groups = [(m, lab, c) for m, lab, c in groups if m.any()]

edges = np.linspace(norms.min(), norms.max(), N_BINS + 1)
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]

fig = go.Figure()
for mask, label, color in groups:
    g = norms[mask]
    name = f"{label} (n={mask.sum():,}, med {np.median(g):.3g})"
    if mask.sum() >= 5:
        counts, _ = np.histogram(g, bins=edges)
        fig.add_bar(x=centers, y=counts, width=width, name=name,
                    marker_color=color, opacity=0.6, marker_line_width=0,
                    hovertemplate="‖h‖ ≈ %{x:.4g}<br>count %{y}<extra>" + label + "</extra>")
    else:  # too few samples for bars — vertical line(s) + a legend entry
        for v in g:
            fig.add_vline(x=float(v), line_color=color, line_width=1.5, opacity=0.8)
        fig.add_scatter(x=[None], y=[None], mode="lines",
                        line=dict(color=color, width=1.5), name=name)

fig.update_layout(
    barmode="overlay", template="plotly_white", height=650,
    title=("pile_4l — residual-stream norm after MLP 2 (h.1 output) by position category<br>"
           f"<sup>{N_ROWS} cached Pile rows × 512 = {len(norms):,} positions; raw counts, "
           f"{N_BINS} bins; drag to zoom, double-click to reset</sup>"),
    xaxis_title="‖h‖", yaxis_title="count",
    legend=dict(x=0.55, y=0.97),
    updatemenus=[dict(
        type="buttons", direction="right", x=0, xanchor="left", y=1.08, yanchor="bottom",
        buttons=[
            dict(label="linear y", method="relayout", args=[{"yaxis.type": "linear"}]),
            dict(label="log y", method="relayout", args=[{"yaxis.type": "log"}]),
        ],
    )],
)

out = HERE.parent / "resid_norms_mlp2_interactive.html"
fig.write_html(out, include_plotlyjs=True)
print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
