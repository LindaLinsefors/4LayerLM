"""Independent re-verification of rope_report.md (2026-09-18).

1. NLL of sink45/46 under configured (base 1e4) vs fitted RoPE on FRESH rows
   64:128 -- the fit used rows 0:48 (train) / 48:64 (val), so these rows touch
   neither -- plus pile_4l on the same rows for reference.
2. The local Torch port (corrected_rope=False) vs the cached official JAX
   loader reference (cache/jax_reference.npz): per-row NLL + strided logits.

Local GPU, ~2 min.  Results 2026-09-18:
  sink45: configured 7.806, fitted-avg 2.504, fitted-own 2.502
  sink46: configured 7.620, fitted-avg 2.499, fitted-own 2.497
  pile_4l: 2.553
  port 8.1054 vs JAX 8.1082, per-row max |diff| 0.008
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from fit_rope_freqs import forward_nll  # noqa: E402
from load import load_pile_4l, load_sink  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = HERE / "cache"
BS = 4

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)
fresh = torch.stack([r[:512] for r in rows[64:128]])

p = np.arange(64)
spectra = {
    "configured base 1e4": -(p / 64) * np.log(1e4),
    "fitted seed-avg": np.load(CACHE / "fitted_freqs_avg.npz")["log_inv_freq"],
    "fitted 45": np.load(CACHE / "fitted_freqs_45.npz")["log_inv_freq"],
    "fitted 46": np.load(CACHE / "fitted_freqs_46.npz")["log_inv_freq"],
}


def batched_nll(model, spec):
    t = torch.tensor(spec, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return float(np.mean([forward_nll(model, fresh[i:i + BS].to(DEVICE), t).item()
                              for i in range(0, len(fresh), BS)]))


print("=== 1. NLL on fresh rows 64:128 (never used in the fit) ===")
for seed in (45, 46):
    m, _ = load_sink(seed)
    m = m.to(DEVICE).eval()
    print(f"sink{seed}:  " + "  ".join(
        f"{n}: {batched_nll(m, spectra[n]):.4f}"
        for n in ("configured base 1e4", "fitted seed-avg", f"fitted {seed}")))
    del m

m4, _, _ = load_pile_4l()
m4 = m4.to(DEVICE).eval()
vals = []
with torch.no_grad():
    for i in range(0, len(fresh), BS):
        b = fresh[i:i + BS].to(DEVICE)
        lo = m4(b)
        vals.append(F.cross_entropy(
            lo[:, :-1].reshape(-1, lo.shape[-1]), b[:, 1:].reshape(-1)).item())
print(f"pile_4l reference: {np.mean(vals):.4f}")
del m4

print("\n=== 2. Torch port (as configured) vs official JAX loader reference ===")
ref = np.load(CACHE / "jax_reference.npz")
m, _ = load_sink(45, corrected_rope=False)
m = m.to(DEVICE).eval()
batch = torch.stack([r[:512] for r in rows[:int(ref["n_rows"])]])
row_nll, logits0 = [], None
with torch.no_grad():
    for i in range(0, len(batch), BS):
        b = batch[i:i + BS].to(DEVICE)
        lo = m(b)
        nll = -lo.log_softmax(-1)[:, :-1].gather(-1, b[:, 1:, None])[..., 0]
        row_nll += nll.mean(dim=1).tolist()
        if i == 0:
            logits0 = lo[0, ::int(ref["stride"])].float().cpu().numpy()
row_nll = np.array(row_nll)
print(f"port mean NLL {row_nll.mean():.4f}  vs JAX ref {float(ref['mean_nll']):.4f}")
print("per-row |diff| max:", np.abs(row_nll - ref["row_nll"]).max())
d = np.abs(logits0 - ref["logits_row0"].astype(np.float32))
print(f"strided logits row0: max|diff| {d.max():.4f}  mean|diff| {d.mean():.5f}")
