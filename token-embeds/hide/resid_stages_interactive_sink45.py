"""resid_stages_interactive.py replicated for the sink seed 45 model
(t-87f91319, untied lm_head + learned attention sinks; loaded via
load_sink(45), which installs the numerically recovered training-time RoPE —
see sink-models/rope_report.md).

Because the head is untied, the embedding (wte) and unembedding (lm_head) give
two different PCA bases; the same residual-stream statistics are rendered once
in each. Basis: PCA of the frequent-token rows (count >= 7 over the 4,000
cached Pile rows, missing tokens thereby excluded) of the chosen matrix,
mean-centered on that group; sign fixed so the frequent-token mean projection
is >= 0 — identical to the pile_4l convention.

The model is run on the same 200 cached Pile rows x 512 positions; captured
after the embedding, each attention, each MLP, and the final norm. Panels per
stage: variance along each PC (log y), |mean projection| (log y), signed mean
projection.

Outputs: token-embeds/resid_stages_interactive_sink45_emb.html
         token-embeds/resid_stages_interactive_sink45_unemb.html
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
N_PILE_ROWS = 200


def pc_basis(mat: np.ndarray, frequent: np.ndarray) -> np.ndarray:
    """(d, d) matrix whose columns are the frequent-token PCs of `mat`
    (vocab, d), sign fixed so the frequent-token mean projection is >= 0."""
    x = mat[frequent] - mat[frequent].mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    mean_proj = (mat[frequent] @ vt.T).mean(axis=0)
    return (vt * np.where(mean_proj >= 0, 1.0, -1.0)[:, None]).T


def stage_stats(model, rows: list[torch.Tensor], bases: dict[str, np.ndarray],
                batch_size: int):
    """One forward sweep; accumulate mean/variance of the residual stream along
    each PC of every basis. Returns {basis_name: (stage_names, mean, var)}."""
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

    V = {name: torch.tensor(b, dtype=torch.float32, device=DEVICE)
         for name, b in bases.items()}
    d = next(iter(V.values())).shape[0]
    s = {name: np.zeros((len(stages), d)) for name in bases}
    q = {name: np.zeros((len(stages), d)) for name in bases}
    n = 0

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = torch.stack(rows[start : start + batch_size]).to(DEVICE)
            model(batch)
            hs = [model.wte(batch)]
            for l in range(n_layer):
                hs.append(hs[-1] + caps[f"attn{l}"])
                hs.append(caps[f"block{l}"])
            hs.append(caps["ln_f"])
            for si, h in enumerate(hs):
                h = h.reshape(-1, d).float()
                for name in bases:
                    p = (h @ V[name]).double()
                    s[name][si] += p.sum(dim=0).cpu().numpy()
                    q[name][si] += (p ** 2).sum(dim=0).cpu().numpy()
            n += batch.numel()
    for h in hooks:
        h.remove()

    out = {}
    for name in bases:
        mean = s[name] / n
        out[name] = (stages, mean, q[name] / n - mean ** 2)
    return out, n


model, _ = load_sink(45)
emb = model.wte.weight.detach().float().numpy()
unemb = model.lm_head.weight.detach().float().numpy()

all_rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                      map_location="cpu", weights_only=True)
counts = np.bincount(torch.stack(all_rows).flatten().numpy(), minlength=len(emb))
frequent = counts >= 7
rows = [r[:512] for r in all_rows[:N_PILE_ROWS]]

bases = {"emb": pc_basis(emb, frequent), "unemb": pc_basis(unemb, frequent)}
stats, n_tok = stage_stats(model, rows, bases, batch_size=16)
del model

LABEL = {"emb": "embedding (wte)", "unemb": "unembedding (lm_head)"}
for variant in ("emb", "unemb"):
    stages, mean, var = stats[variant]
    d = mean.shape[1]
    pcs = np.arange(1, d + 1)
    lab = LABEL[variant]

    colors = sample_colorscale("Turbo", np.linspace(0.08, 0.92, len(stages)))
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"variance of the residual stream along each {lab} PC direction, per stage",
            f"|mean projection| onto each {lab} PC",
            f"mean projection onto each {lab} PC (sign: frequent-token mean ≥ 0)",
        ),
        vertical_spacing=0.09,
    )
    for stage, m, vr, color in zip(stages, mean, var, colors):
        fig.add_scatter(
            x=pcs, y=vr, mode="lines", row=1, col=1,
            name=stage, legendgroup=stage,
            line=dict(color=color, width=1.5),
            hovertemplate="PC %{x}<br>variance %{y:.3e}<extra>" + stage + "</extra>",
        )
        fig.add_scatter(
            x=pcs, y=np.abs(m), mode="lines", row=2, col=1,
            name=stage, legendgroup=stage, showlegend=False,
            line=dict(color=color, width=1.5),
            hovertemplate="PC %{x}<br>|mean proj| %{y:.3e}<extra>" + stage + "</extra>",
        )
        fig.add_scatter(
            x=pcs, y=m, mode="lines", row=3, col=1,
            name=stage, legendgroup=stage, showlegend=False,
            line=dict(color=color, width=1.5),
            hovertemplate="PC %{x}<br>mean proj %{y:.4f}<extra>" + stage + "</extra>",
        )

    fig.update_yaxes(type="log", title_text="variance along PC", row=1, col=1)
    fig.update_yaxes(type="log", title_text="|mean projection|", row=2, col=1)
    fig.update_yaxes(title_text="mean projection", row=3, col=1)
    fig.update_xaxes(title_text=f"{lab} principal component index (frequent-token PCA)",
                     row=3, col=1)
    fig.update_layout(
        height=1200, template="plotly_white",
        title=f"sink seed 45 (t-87f91319) — residual stream by stage, in the "
              f"frequent-token {lab}-PCA basis ({n_tok:,} training-data positions, "
              f"d={d}; corrected RoPE — see sink-models/rope_report.md; "
              "drag to zoom, double-click to reset; legend toggles all panels)",
        legend=dict(groupclick="togglegroup"),
    )

    out = HERE.parent / f"resid_stages_interactive_sink45_{variant}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
