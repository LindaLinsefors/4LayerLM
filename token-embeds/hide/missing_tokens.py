"""Compute and save the pile_4l "missing tokens" list (CLAUDE.md 2026-08-27).

Missing tokens = the sharp line at PC1 ~= 0.71 in the count-0 strip: tokens
with zero count in the 2.05M-token sample AND projection > 0.68 onto PC1 of
the frequent-token PCA (sign: frequent mean >= 0). Verified 2026-08-27 to be
tokens with (essentially) zero exposure in the actual training data (very long
whitespace runs, all-caps fragments, mojibake), sitting at the push-away/decay
cone equilibrium; the other count-0 tokens are merely rare (diffuse cloud,
PC1 < 0.62; only 60 tokens fall in the gap).

Output: token-embeds/missing_tokens.npy (int64 token ids, n=304)
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent / "missing_tokens.npy"

emb = load_file(
    ROOT / "prev_paper/models/pile_4layer/target_model_t-9d2b8f02/model_step_99999.safetensors"
)["wte.weight"].float().numpy()
counts = np.bincount(torch.stack(torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True
)).flatten().numpy(), minlength=len(emb))

frequent = counts >= 7
centered = emb[frequent] - emb[frequent].mean(axis=0)
_, _, vt = np.linalg.svd(centered, full_matrices=False)
pc1 = emb @ vt[0]
if pc1[frequent].mean() < 0:
    pc1 = -pc1

missing = np.where((counts == 0) & (pc1 > 0.68))[0]
np.save(OUT, missing)
print(f"saved {OUT}: {len(missing)} missing tokens "
      f"(PC1 range {pc1[missing].min():.3f}..{pc1[missing].max():.3f})")
