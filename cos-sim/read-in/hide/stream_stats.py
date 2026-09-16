"""Per-site residual-stream input statistics for the bias-direction analysis
(cos-sim/read-in/bias_direction_report.md).

For each of the 8 norm sites h.<l>.rms_{1,2} (rms_1 output = the input the
q/k/v read-ins see, rms_2 output = the input c_fc sees, i.e. x = g * h/rms(h)
including the gain), accumulates mean and second moment of x over Pile data:
  "h.<l>.rms_<i>|S1"  (768,)      sum of x            (float64)
  "h.<l>.rms_<i>|S2"  (768,768)   sum of outer(x, x)  (float64)
  "n"                 scalar      number of positions
for both relevant target models:
  cache/stream_stats_pile4l.npz  — t-9d2b8f02 (decomposed by newA)
  cache/stream_stats_sink45.npz  — t-87f91319 (decomposed by C), loaded via
                                   load.load_sink(45) => corrected RoPE.

Positions 0 and 1 (attention-sink sites) and <|endoftext|> tokens are
excluded from the statistics (massive-vector outliers; 0.4% of positions).
500 cached Pile rows x 510 kept positions ~ 255k samples.

Usage: python cos-sim/read-in/hide/stream_stats.py   (~2 min local GPU)
"""
import sys
import numpy as np
import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
import load  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

N_ROWS = 500
BATCH = 8
EOS = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

rows = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")
rows = torch.stack([r[:512] for r in rows[:N_ROWS]])


def run(tag, model):
    model = model.to(DEV).eval()
    sites = [f"h.{l}.rms_{i}" for l in range(4) for i in (1, 2)]
    acc = {s: [torch.zeros(768, dtype=torch.float64, device=DEV),
               torch.zeros(768, 768, dtype=torch.float64, device=DEV)]
           for s in sites}
    n_tot = 0
    mask_state = {}

    def hook(site):
        def fn(mod, inp, out):
            x = out.reshape(-1, 768)[mask_state["m"]].double()
            acc[site][0] += x.sum(0)
            acc[site][1] += x.T @ x
        return fn

    handles = [getattr(model.h[l], f"rms_{i}").register_forward_hook(
        hook(f"h.{l}.rms_{i}")) for l in range(4) for i in (1, 2)]

    with torch.no_grad():
        for b in range(0, len(rows), BATCH):
            tok = rows[b:b + BATCH].to(DEV)
            keep = tok != EOS
            keep[:, :2] = False                      # sink positions 0, 1
            mask_state["m"] = keep.reshape(-1)
            model(tok)
            n_tot += int(keep.sum())
    for h in handles:
        h.remove()

    out = {"n": np.array(n_tot)}
    for s in sites:
        out[f"{s}|S1"] = acc[s][0].cpu().numpy()
        out[f"{s}|S2"] = acc[s][1].cpu().numpy()
    np.savez_compressed(CACHE / f"stream_stats_{tag}.npz", **out)
    print(f"{tag}: {n_tot} positions -> stream_stats_{tag}.npz")
    model.to("cpu")


run("pile4l", load._load_target_model(
    load.PILE_4L / "target_model_t-9d2b8f02", "model_step_99999.safetensors"))
run("sink45", load.load_sink(45)[0])
