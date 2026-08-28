"""Cosine-similarity grid between the LAST PC (smallest-variance direction) of
the residual stream at every stage (after embedding, after each attention,
after each MLP, after the final norm — same data rows as final_acts.py /
resid_stages_interactive.py) plus two reference directions from the
frequent/alive-token embedding PCA: PC1 (the frequency axis) and PC_last (the
bias direction).

Per stage the full covariance is accumulated batchwise (d x d, nothing large
cached); the stage's last PC is the eigenvector of its smallest eigenvalue.
Values shown are |cos| (PC signs are arbitrary).

Output: token-embeds/last_pc_cosines.png (one heatmap per model)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l, load_simple_2l

OUT = HERE.parent / "last_pc_cosines.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_PILE_ROWS = 200
N_SIMPLE_STORIES = 360


def stage_last_pcs(model, rows: list[torch.Tensor], batch_size: int):
    """Smallest-variance direction of the residual stream at every stage.
    Returns (stage_names, (n_stages, d) array of unit vectors)."""
    model = model.to(DEVICE)
    n_layer = len(model.h)
    stages = (["after embedding"]
              + [f"after {kind} {l + 1}" for l in range(n_layer) for kind in ("attn", "MLP")]
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

    d = model.config.n_embd
    s = torch.zeros(len(stages), d, dtype=torch.float64, device=DEVICE)
    Q = torch.zeros(len(stages), d, d, dtype=torch.float64, device=DEVICE)
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
                hs = [model.wte(batch)]
                for l in range(n_layer):
                    hs.append(hs[-1] + caps[f"attn{l}"])
                    hs.append(caps[f"block{l}"])
                hs.append(caps["ln_f"])
                for si, h in enumerate(hs):
                    p = h.reshape(-1, d).double()
                    s[si] += p.sum(dim=0)
                    Q[si] += p.T @ p
                n += len(chunk) * length
    for h in hooks:
        h.remove()

    last_pcs = np.empty((len(stages), d))
    for si in range(len(stages)):
        mu = (s[si] / n).cpu().numpy()
        cov = (Q[si] / n).cpu().numpy() - np.outer(mu, mu)
        _, v = np.linalg.eigh(cov)
        last_pcs[si] = v[:, 0]
    return stages, last_pcs


def emb_pc1_last(name: str, emb: np.ndarray):
    if name == "pile_4l":
        rows = torch.stack(torch.load(
            ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
        ))
        alive = np.bincount(rows.flatten().numpy(), minlength=len(emb)) >= 7
    else:
        table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
        counts = np.zeros(len(emb), dtype=np.int64)
        counts[table["id"].to_numpy()] = table["freq"].to_numpy()
        alive = counts >= 10
    x = emb[alive] - emb[alive].mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[0], vt[-1]


fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.5), width_ratios=[12, 8])
for ax, (name, loader, n_rows, batch_size) in zip(axes, [
    ("pile_4l", load_pile_4l, N_PILE_ROWS, 16),
    ("simple_2l", load_simple_2l, N_SIMPLE_STORIES, 64),
]):
    model, _, _ = loader()
    emb = model.wte.weight.detach().float().numpy()

    if name == "pile_4l":
        rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                          map_location="cpu", weights_only=True)[:n_rows]
    else:
        rows = torch.load(ROOT / "context-loss/hide/cache/simple_stories.pt",
                          map_location="cpu", weights_only=True)[:n_rows]
    rows = [r[:512] for r in rows]

    stages, last_pcs = stage_last_pcs(model, rows, batch_size)
    del model
    pc1, pc_last = emb_pc1_last(name, emb)

    labels = ["emb PC_last"] \
             + [st.replace("after ", "") + " (last PC)" for st in stages] \
             + ["emb PC1"]
    dirs = np.vstack([pc_last, last_pcs, pc1])
    cos = np.abs(dirs @ dirs.T)
    k = len(labels)

    ax.imshow(cos, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cos[i, j] > 0.6 else "#333333")
    ax.set_xticks(range(k), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k), labels, fontsize=8)
    ax.set_title(f"{name}  (|cos|; stream last PCs over "
                 f"{sum(len(r) for r in rows):,} training positions)", fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("Cosine similarity: per-stage smallest-variance residual-stream directions "
             "vs token-embedding PC1 / PC_last", y=0.98)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
