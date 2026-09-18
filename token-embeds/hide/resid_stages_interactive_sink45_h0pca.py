"""resid_stages_interactive_sink45 variant whose PCA basis is the residual
stream itself, at a SELECTABLE stage (dropdown in the page): after embedding,
after each attention, after each MLP, or after the final norm.

One forward sweep over all 4,000 cached Pile rows x 512 positions (2.048M
tokens) accumulates, per stage, the sufficient statistics sum(h) and
sum(h h^T) in float64. From these everything is exact, no second pass:
the stage-i covariance C_i eigendecomposes into that stage's PCA basis, and
the mean/variance of any stage j along any basis-i PC are mu_j.v and v^T C_j v.
Consequently, with basis = stage i, the stage-i variance curve is exactly the
descending eigenvalue spectrum of C_i.

The "after embedding" basis equals the token-frequency-weighted embedding PCA
(no positional embedding, so the stage-0 stream is wte[token], weighted by
corpus counts over the same 4,000 rows). Missing tokens never appear and drop
out automatically. Sign convention (all bases): the basis stage's own mean
projection is >= 0 (mu_i.v >= 0) — NOTE this replaces the earlier
frequent-token-mean convention of the single-basis version of this page, so
panel-3 signs for the stage-0 basis can flip on individual PCs vs the old file.

Sink seed 45 loaded via load_sink(45) (corrected RoPE — see
sink-models/rope_report.md). Sweep ~5 min local GPU, cached in
hide/cache/stream_grams_sink45.npz (delete to recompute).

Output: token-embeds/resid_stages_interactive_sink45_h0pca.html
"""

import sys
from pathlib import Path

import numpy as np
import torch
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_sink

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = HERE / "cache" / "stream_grams_sink45.npz"
BATCH = 16


def sweep_grams():
    """Forward all 4,000 cached rows; per stage accumulate S = sum h (d,) and
    G = sum h h^T (d, d) in float64. Returns (stage_names, S, G, n)."""
    model, _ = load_sink(45)
    model = model.to(DEVICE)
    n_layer = len(model.h)
    stages = (["after embedding"]
              + [f"after {kind} {l + 1}" for l in range(n_layer) for kind in ("attention", "MLP")]
              + ["after final norm"])

    caps: dict[str, torch.Tensor] = {}
    hooks = []
    for l, block in enumerate(model.h):
        hooks.append(block.attn.register_forward_hook(
            lambda _m, _i, out, l=l: caps.__setitem__(f"attn{l}", out)))
        hooks.append(block.register_forward_hook(
            lambda _m, _i, out, l=l: caps.__setitem__(f"block{l}", out)))
    hooks.append(model.ln_f.register_forward_hook(
        lambda _m, _i, out: caps.__setitem__("ln_f", out)))

    rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)
    rows = [r[:512] for r in rows]
    d = model.wte.weight.shape[1]
    S = np.zeros((len(stages), d))
    G = np.zeros((len(stages), d, d))
    n = 0

    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            batch = torch.stack(rows[start : start + BATCH]).to(DEVICE)
            model(batch)
            hs = [model.wte(batch)]
            for l in range(n_layer):
                hs.append(hs[-1] + caps[f"attn{l}"])
                hs.append(caps[f"block{l}"])
            hs.append(caps["ln_f"])
            for si, h in enumerate(hs):
                h = h.reshape(-1, d).float()
                S[si] += h.sum(dim=0).double().cpu().numpy()
                G[si] += (h.T @ h).double().cpu().numpy()
            n += batch.numel()
    for hk in hooks:
        hk.remove()
    return stages, S, G, n


if CACHE.exists():
    z = np.load(CACHE, allow_pickle=False)
    stages, S, G, n = list(z["stages"]), z["S"], z["G"], int(z["n"])
else:
    stages, S, G, n = sweep_grams()
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, stages=np.array(stages), S=S, G=G, n=n)
    print(f"cached sweep stats -> {CACHE}")

n_stage = len(stages)
d = S.shape[1]
mu = S / n
C = G / n - np.einsum("id,ie->ide", mu, mu)

# per-stage PCA bases (columns, eigenvalues descending; sign: own mean proj >= 0)
bases = []
for i in range(n_stage):
    evals, evecs = np.linalg.eigh(C[i])
    evecs = evecs[:, ::-1]
    sign = np.where(mu[i] @ evecs >= 0, 1.0, -1.0)
    bases.append(evecs * sign[None, :])
    # exactness check: own-stage variance along PC k == eigenvalue k
    own_var = np.einsum("dk,de,ek->k", bases[i], C[i], bases[i])
    assert np.allclose(own_var, evals[::-1], rtol=1e-8, atol=1e-12)

# mean/var of every stage along every basis: (basis, stage, pc)
mean = np.einsum("jd,bdk->bjk", mu, np.stack(bases))
var = np.einsum("bdk,jde,bek->bjk", np.stack(bases), C, np.stack(bases))

pcs = np.arange(1, d + 1)
colors = sample_colorscale("Turbo", np.linspace(0.08, 0.92, n_stage))
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        "variance of the residual stream along each PC of the selected basis, per stage",
        "|mean projection| onto each PC of the selected basis",
        "mean projection onto each PC (sign: basis stage's own mean projection ≥ 0)",
    ),
    vertical_spacing=0.09,
)
for bi in range(n_stage):
    vis = bi == 0
    for stage, m, vr, color in zip(stages, mean[bi], var[bi], colors):
        common = dict(name=stage, legendgroup=f"{bi}:{stage}", visible=vis,
                      line=dict(color=color, width=1.5))
        fig.add_scatter(
            x=pcs, y=vr, mode="lines", row=1, col=1, **common,
            hovertemplate="PC %{x}<br>variance %{y:.3e}<extra>" + stage + "</extra>",
        )
        fig.add_scatter(
            x=pcs, y=np.abs(m), mode="lines", row=2, col=1, showlegend=False, **common,
            hovertemplate="PC %{x}<br>|mean proj| %{y:.3e}<extra>" + stage + "</extra>",
        )
        fig.add_scatter(
            x=pcs, y=m, mode="lines", row=3, col=1, showlegend=False, **common,
            hovertemplate="PC %{x}<br>mean proj %{y:.4f}<extra>" + stage + "</extra>",
        )

buttons = []
for bi, bstage in enumerate(stages):
    vis = [False] * (3 * n_stage * n_stage)
    vis[3 * n_stage * bi : 3 * n_stage * (bi + 1)] = [True] * (3 * n_stage)
    buttons.append(dict(label=f"basis: {bstage}", method="restyle",
                        args=[{"visible": vis}]))

fig.update_yaxes(type="log", title_text="variance along PC", row=1, col=1)
fig.update_yaxes(type="log", title_text="|mean projection|", row=2, col=1)
fig.update_yaxes(title_text="mean projection", row=3, col=1)
fig.update_xaxes(title_text="principal component index of the selected-stage stream PCA",
                 row=3, col=1)
fig.update_layout(
    height=1200, template="plotly_white", margin=dict(t=110, b=100),
    title=dict(
        text="sink seed 45 (t-87f91319) — residual stream by stage, in the "
             "stream-PCA basis of the dropdown-selected stage "
             f"({n:,} training positions, d={d}; corrected RoPE)",
        x=1.0, xanchor="right",
    ),
    legend=dict(groupclick="togglegroup"),
    updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True,
        x=0.0, xanchor="left", y=1.10, yanchor="top",
    )],
    annotations=list(fig.layout.annotations) + [dict(
        text="Bases and statistics come from the same positions, so with basis stage = "
             "shown stage the variance curve is exactly the eigenvalue spectrum. "
             "Sign: basis stage's own mean projection ≥ 0. Drag to zoom, double-click "
             "to reset; legend toggles all panels. RoPE: see sink-models/rope_report.md.",
        x=0.0, xref="paper", y=-0.075, yref="paper",
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=11, color="#555"), align="left",
    )],
)

out = HERE.parent / "resid_stages_interactive_sink45_h0pca.html"
fig.write_html(out, include_plotlyjs=True)
print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
