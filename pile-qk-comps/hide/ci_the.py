"""Do the L0 q_proj "the"-components activate on every "the" but only get
causal importance on the first one?

Runs 32 cached Pile rows through the decomposition: per-token CI (lower_leaky)
and per-token subcomponent activation a_c = ||U_c|| (V_c . phi) for components
38, 212, 270, 345 of h.0.attn.q_proj.  Breaks both down by the occurrence rank
of "the" within the visible document segment (rank restarts after EOS), and
computes pairwise similarity (Pearson r over tokens) of |a_c| and CI patterns.

Run with the 3.13 venv:
  param-decomp-vpd\\.venv\\Scripts\\python.exe pile-qk-comps/hide/ci_the.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from load import load_tokenizer

MODULE = "h.0.attn.q_proj"
COMPS = [38, 212, 270, 345]
N_ROWS = 32
BATCH = 8
EOS = 0
CKPT = (ROOT / "prev_paper" / "models" / "pile_4layer"
        / "vpd_decomposition_s-55ea3f9b" / "model_400000.pth")

from param_decomp.models.component_model import ComponentModel

tok = load_tokenizer("pile_4l")
rows = torch.load(ROOT / "context-loss" / "hide" / "cache" / "pile_rows.pt")[:N_ROWS]
rows = torch.stack([r[:512] for r in rows])

model = ComponentModel.from_pretrained(str(CKPT))
model.eval()

V = dict(model.named_parameters())[f"_components.{MODULE.replace('.', '-')}.V"].detach()
U = dict(model.named_parameters())[f"_components.{MODULE.replace('.', '-')}.U"].detach()
u_norm = U.norm(dim=1)                                     # (C,)

ci_all, act_all = [], []
with torch.no_grad():
    for i in range(0, N_ROWS, BATCH):
        batch = rows[i:i + BATCH]
        out = model(batch, cache_type="input")
        ci = model.calc_causal_importances(out.cache, sampling="continuous")
        ci_all.append(ci.lower_leaky[MODULE][:, :, COMPS])
        phi = out.cache[MODULE]                            # (B, 512, d_in)
        act_all.append((phi @ V[:, COMPS]) * u_norm[COMPS])
ci_v = torch.cat(ci_all).reshape(-1, len(COMPS)).numpy()    # (N*512, 4)
act_v = torch.cat(act_all).reshape(-1, len(COMPS)).abs().numpy()

# "the" occurrence rank within the document segment visible in the row
flat_ids = rows.reshape(-1).numpy()
tok_strs = [tok.convert_tokens_to_string([t]) for t in tok.convert_ids_to_tokens(rows.reshape(-1).tolist())]
is_the = np.array([s.strip().lower() == "the" for s in tok_strs])
rank = np.zeros(len(flat_ids), dtype=int)                  # 0 = not "the"
for r in range(N_ROWS):
    count = 0
    for j in range(512):
        k = r * 512 + j
        if flat_ids[k] == EOS:
            count = 0
        elif is_the[k]:
            count += 1
            rank[k] = count

print(f"{MODULE} components {COMPS} over {N_ROWS} rows "
      f"({is_the.sum()} 'the' tokens, {(rank == 1).sum()} first-in-doc)")

print("\nmean CI / mean |a| / frac(CI>0.1) by 'the' occurrence rank in doc segment:")
hdr = "rank    n   " + "".join(f"| c{c}: CI    |a|   f>.1 " for c in COMPS)
print(hdr)
for lo, hi, name in [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
                     (6, 10, "6-10"), (11, 999, ">10")]:
    m = (rank >= lo) & (rank <= hi)
    line = f"{name:>4} {m.sum():5d}  "
    for k in range(len(COMPS)):
        line += (f"|  {ci_v[m, k].mean():.3f} {act_v[m, k].mean():6.2f} "
                 f"{(ci_v[m, k] > 0.1).mean():.2f} ")
    print(line)
m = ~is_the
line = f"non-the {m.sum():4d}  "
for k in range(len(COMPS)):
    line += (f"|  {ci_v[m, k].mean():.3f} {act_v[m, k].mean():6.2f} "
             f"{(ci_v[m, k] > 0.1).mean():.2f} ")
print(line)

def corr_matrix(X: np.ndarray, label: str) -> None:
    C = np.corrcoef(X.T)
    print(f"\npairwise Pearson r of {label}:")
    print("        " + "  ".join(f"c{c:>4}" for c in COMPS))
    for i, c in enumerate(COMPS):
        print(f"  c{c:>4} " + "  ".join(f"{C[i, j]:5.2f}" for j in range(len(COMPS))))

corr_matrix(act_v, "|activation| (all tokens)")
corr_matrix(ci_v, "CI (all tokens)")
the_mask = is_the
corr_matrix(ci_v[the_mask], "CI ('the' tokens only)")
corr_matrix(act_v[the_mask], "|activation| ('the' tokens only)")
