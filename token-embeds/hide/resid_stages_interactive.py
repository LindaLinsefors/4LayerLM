"""Interactive (Plotly) view of the residual stream at every stage of the
network, expressed in the token-embedding PCA basis. One self-contained HTML
per model, same style as pca_interactive.py.

X-axis: the embedding PC directions (PCA of the frequent/alive-token
embeddings, mean-centered on that group; sign fixed so the frequent/alive
token mean projection is >= 0 — identical to pca.py / pca_interactive.py).

The model is run on representative training data (same rows as final_acts.py:
200 cached Pile rows x 512 / first 360 cached stories), and the residual
stream is captured after the embedding, after each attention sublayer, after
each MLP sublayer, and after the final norm. For each stage:

  panel 1 (log y): variance of the activations along each embedding PC
  panel 2 (log y): |mean projection| of the raw activations onto each PC
  panel 3:         mean projection (signed)

Sums are accumulated batchwise in projection space, so nothing large is cached.

Outputs: token-embeds/resid_stages_interactive_{pile_4l,simple_2l}.html
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from safetensors.torch import load_file

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l, load_simple_2l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_PILE_ROWS = 200
N_SIMPLE_STORIES = 360


def emb_pc_basis(name: str, emb: np.ndarray) -> np.ndarray:
    """(d, d) matrix whose columns are the frequent/alive-token embedding PCs,
    sign fixed so the frequent/alive mean projection is >= 0."""
    if name == "pile_4l":
        rows = torch.stack(torch.load(
            ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
        ))
        counts = np.bincount(rows.flatten().numpy(), minlength=len(emb))
        alive = counts >= 7
    else:
        table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
        counts = np.zeros(len(emb), dtype=np.int64)
        counts[table["id"].to_numpy()] = table["freq"].to_numpy()
        alive = counts >= 10
    x = emb[alive] - emb[alive].mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    alive_mean = (emb[alive] @ vt.T).mean(axis=0)
    return (vt * np.where(alive_mean >= 0, 1.0, -1.0)[:, None]).T


def stage_stats(model, rows: list[torch.Tensor], basis: np.ndarray, batch_size: int):
    """Accumulate mean and variance of the residual stream along each
    embedding-PC direction, at every stage. Returns (stage_names, mean, var)
    with mean/var of shape (n_stages, d)."""
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

    V = torch.tensor(basis, dtype=torch.float32, device=DEVICE)  # (d, d), columns = PCs
    d = V.shape[0]
    s = np.zeros((len(stages), d))   # sum of projections
    q = np.zeros((len(stages), d))   # sum of squared projections
    n = 0

    with torch.no_grad():
        by_len: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by_len.setdefault(len(r), []).append(i)
        for length, idxs in sorted(by_len.items()):
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start : start + batch_size]
                batch = torch.stack([rows[i] for i in chunk]).to(DEVICE)
                model(batch)
                x = model.wte(batch)
                hs = [x]
                for l in range(n_layer):
                    hs.append(hs[-1] + caps[f"attn{l}"])   # after attention l
                    hs.append(caps[f"block{l}"])           # after MLP l (block output)
                hs.append(caps["ln_f"])
                for si, h in enumerate(hs):
                    p = (h.reshape(-1, d).float() @ V).double()
                    s[si] += p.sum(dim=0).cpu().numpy()
                    q[si] += (p ** 2).sum(dim=0).cpu().numpy()
                n += len(chunk) * length
    for h in hooks:
        h.remove()

    mean = s / n
    var = q / n - mean ** 2
    return stages, mean, var


for name, loader, n_rows, batch_size in [
    ("pile_4l", load_pile_4l, N_PILE_ROWS, 16),
    ("simple_2l", load_simple_2l, N_SIMPLE_STORIES, 64),
]:
    model, _, _ = loader()
    emb = model.wte.weight.detach().float().numpy()
    basis = emb_pc_basis(name, emb)

    if name == "pile_4l":
        rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                          map_location="cpu", weights_only=True)[:n_rows]
    else:
        rows = torch.load(ROOT / "context-loss/hide/cache/simple_stories.pt",
                          map_location="cpu", weights_only=True)[:n_rows]
    rows = [r[:512] for r in rows]  # n_ctx = 512 (pile rows are 513, a few stories longer)

    stages, mean, var = stage_stats(model, rows, basis, batch_size)
    del model
    d = mean.shape[1]
    pcs = np.arange(1, d + 1)
    n_tok = sum(len(r) for r in rows)

    colors = sample_colorscale("Turbo", np.linspace(0.08, 0.92, len(stages)))
    base = "frequent" if name == "pile_4l" else "alive"
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "variance of the residual stream along each embedding-PC direction, per stage",
            "|mean projection| onto each embedding PC",
            f"mean projection onto each embedding PC (sign: {base}-token embedding mean ≥ 0)",
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
    fig.update_xaxes(title_text=f"embedding principal component index ({base}-token PCA)", row=3, col=1)
    fig.update_layout(
        height=1200, template="plotly_white",
        title=f"{name} — residual stream by stage, in the {base}-token embedding-PCA basis "
              f"({n_tok:,} training-data positions, d={d}; "
              "drag to zoom, double-click to reset; legend toggles both panels)",
        legend=dict(groupclick="togglegroup"),
    )

    out = HERE.parent / f"resid_stages_interactive_{name}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
