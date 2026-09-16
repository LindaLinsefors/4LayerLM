"""Histograms of residual-stream norms at every stage of pile_4l, split by
position category:

  * position 0 of the window (the first-token sink site)
  * <|endoftext|> at position 0 (both at once) — included only if present
  * <|endoftext|> mid-sequence
  * everything else

Stages: after embedding, after each attention sublayer, after each MLP
sublayer, and after the final norm — 10 panels. Only per-position norms are
kept (no vector cache). Histograms are probability densities (density=True)
so the tiny groups are comparable to the bulk. Each group's full min-max norm
range is drawn as a horizontal line (with end caps) near the top of the panel
in the group's color — bins spanning outliers are often too short to see at
density scale, the range lines show how far each group really extends.

Data: first N_ROWS cached Pile rows (context-loss/hide/cache/pile_rows.pt),
truncated to n_ctx = 512. simple_2l is skipped: its cached stories contain no
[EOS] positions (the loader encodes stories without appending it).

Output: endoftext-pos0/resid_stage_norms.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 1000        # x 512 positions; ~0.4 EOS positions per row
BATCH_SIZE = 16
EOS_ID = 0


def stage_norms(model, rows: list[torch.Tensor]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Run the model on `rows`; return (stage_names, norms (n_stages, N),
    token_ids (N,)) with one norm per stage and token position."""
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

    norms = [[] for _ in stages]
    token_ids = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = torch.stack(rows[start : start + BATCH_SIZE]).to(DEVICE)
            model(batch)
            hs = [model.wte(batch)]
            for l in range(n_layer):
                hs.append(hs[-1] + caps[f"attn{l}"])   # after attention l
                hs.append(caps[f"block{l}"])           # after MLP l (block output)
            hs.append(caps["ln_f"])
            for si, h in enumerate(hs):
                norms[si].append(h.float().norm(dim=-1).flatten().cpu())
            token_ids.append(batch.cpu().flatten())
    for h in hooks:
        h.remove()
    return stages, np.stack([torch.cat(n).numpy() for n in norms]), torch.cat(token_ids).numpy()


rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:512] for r in rows]  # rows are 513 tokens, n_ctx = 512

model, _, _ = load_pile_4l()
stages, norms, token_ids = stage_norms(model, rows)
del model

pos = np.tile(np.arange(512), N_ROWS)
is_eos = token_ids == EOS_ID
groups = [  # (mask, label, color) — draw order: bulk first, small groups on top
    (~is_eos & (pos > 0), "other positions", "tab:blue"),
    (is_eos & (pos > 0), "<|endoftext|> (mid-sequence)", "tab:red"),
    (~is_eos & (pos == 0), "position 0", "tab:orange"),
    (is_eos & (pos == 0), "<|endoftext|> at position 0", "tab:purple"),
]
groups = [(m, lab, c) for m, lab, c in groups if m.any()]
for m, lab, _ in groups:
    print(f"{lab}: {m.sum():,} positions")

def make_figure(ignore_tallest: int, fname: str, subtitle: str) -> None:
    """One 5x2 figure. Per panel, the y-limit fits all histogram groups except
    the `ignore_tallest` ones with the highest peak density (their bars are
    clipped; the vline group never counts)."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)
    for ax, stage, n in zip(axes.flat, stages, norms):
        bins = np.linspace(n.min(), n.max(), 80)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        peaks = []
        for gi, (mask, label, color) in enumerate(groups):
            g = n[mask]
            if mask.sum() >= 5:
                dens, _, _ = ax.hist(g, bins=bins, density=True, color=color,
                                     alpha=0.6, label=f"{label} (n={mask.sum():,})")
                peaks.append(dens.max())
            else:  # too few samples for a density bar (one sample = 1/binwidth tall)
                for v in g:
                    ax.axvline(v, color=color, alpha=0.8, lw=1.5)
                ax.plot([], [], color=color, lw=1.5, label=f"{label} (n={mask.sum():,})")
            ax.plot([g.min(), g.max()], [0.97 - 0.04 * gi] * 2, color=color,
                    lw=1.5, marker="|", ms=7, transform=trans, clip_on=False)
        if ignore_tallest:
            kept = sorted(peaks)[: max(1, len(peaks) - ignore_tallest)]
            ax.set_ylim(0, 1.1 * kept[-1])
        meds = ", ".join(f"{np.median(n[m]):.3g}" for m, _, _ in groups)
        ax.set_title(f"{stage}   (medians: {meds})", fontsize=10)
        ax.set_xlabel("‖h‖")
        ax.set_ylabel("density")
    axes.flat[0].legend(fontsize=8, loc="upper left")

    fig.suptitle("pile_4l — residual-stream norm by stage and position category\n"
                 f"({N_ROWS} cached Pile rows × 512 = {norms.shape[1]:,} positions; "
                 "probability-density histograms; horizontal lines = each group's "
                 f"min-max range{subtitle})", fontsize=12)
    out = HERE.parent / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


make_figure(0, "resid_stage_norms.png", "")
make_figure(2, "resid_stage_norms_zoom.png",
            ";\ny-axis fits only the two least-tall histogram groups per panel — taller ones are clipped")
for stage, n in zip(stages, norms):
    print(f"{stage}: " + "; ".join(
        f"{lab} med {np.median(n[m]):.2f} [{n[m].min():.2f}, {n[m].max():.2f}]"
        for m, lab, _ in groups))
