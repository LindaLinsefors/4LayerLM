"""Cross-seed validation of the fitted RoPE spectra: does each seed's fitted
spectrum transfer to the other seed's model? If yes, the recovered spectrum is
the shared training-time convention, not a per-model artifact. Also scores the
seed-averaged spectrum and two closed-form candidates."""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from fit_rope_freqs import forward_nll
from load import load_sink

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[48:64]
val = torch.stack([r[:512] for r in rows]).to(DEVICE)

f45 = np.load(HERE / "cache/fitted_freqs_45.npz")["log_inv_freq"] \
    if (HERE / "cache/fitted_freqs_45.npz").exists() \
    else np.load(HERE / "cache/fitted_freqs.npz")["log_inv_freq"]
f46 = np.load(HERE / "cache/fitted_freqs_46.npz")["log_inv_freq"]
favg = (f45 + f46) / 2
np.savez(HERE / "cache/fitted_freqs_avg.npz", log_inv_freq=favg)

p = np.arange(64)
geo = -(p / 64) * np.log(1.6e6)
geo_s = geo + np.log(1.477)

for seed, own, other in [(45, f45, f46), (46, f46, f45)]:
    m, _ = load_sink(seed)
    m = m.to(DEVICE)
    out = []
    for name, v in [("own fit", own), ("other-seed fit", other),
                    ("seed-avg fit", favg), ("geometric 1.6e6", geo),
                    ("1.477*geo 1.6e6", geo_s)]:
        t = torch.tensor(v, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out.append(f"{name}: {forward_nll(m, val, t).item():.4f}")
    print(f"seed {seed}:  " + "  ".join(out))
    del m
