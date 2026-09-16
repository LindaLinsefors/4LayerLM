"""Causal test: is the attention-created position signal what triggers the
MLP-2 massive vector?

At position p the attention output is a softmax mixture over keys 0..p; at
p = 0 it is exactly the self-only output o_proj(v_proj(rms_1(x)))[p] (one key).
Interventions (forward hooks that edit the attention output of layer 1 and/or
layer 2 at one position, everything downstream reruns naturally):

  kill @ 0:    replace attention output at position 0 with the same row's
               attention output at bulk position REF = 300 — position 0 now
               looks (to that sublayer) like a bulk position.
               Variants: kill attn 1 only; kill attn 1 + attn 2.
  graft @ 256: replace attention output at bulk position GRAFT_P = 256 with
               the self-only output for the token that sits there — position
               256 now gets exactly what position 0 mechanically gets.
               Variants: graft attn 1 only; graft attn 1 + attn 2.

Measured at the intervened position, per stage: mean residual norm and mean
projection onto the massive-vector direction u (unit mean of baseline pos-0
after-MLP-2 vectors). If the signal is necessary, kill removes the ~220-norm
vector; if sufficient, graft creates it at position 256.

Data: first N_ROWS cached Pile rows; 5 forward passes (~2 min local GPU).
Output: endoftext-pos0/pos0_patch.png + printed table.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_pile_4l

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 500
BATCH_SIZE = 16
EOS_ID = 0
T = 512
REF = 300          # bulk position whose attention output replaces position 0
GRAFT_P = 256      # bulk position that receives the self-only output

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]
tok0 = np.array([int(r[0]) for r in rows])
tokG = np.array([int(r[GRAFT_P]) for r in rows])
tokR = np.array([int(r[REF]) for r in rows])

model, _, _ = load_pile_4l()
model = model.to(DEVICE)
n_layer = len(model.h)
stages = (["after embedding"]
          + [f"after {kind} {l + 1}" for l in range(n_layer) for kind in ("attention", "MLP")]
          + ["after final norm"])


def patch_hook(kind: str, pos: int):
    def hook(mod, inputs, out):
        out = out.clone()
        if kind == "kill":
            out[:, pos] = out[:, REF]
        else:  # graft: the self-only (single-key softmax) output for this token
            out[:, pos] = mod.o_proj(mod.v_proj(inputs[0][:, pos]))
        return out
    return hook


def run(patches: list[tuple[int, str, int]]) -> np.ndarray:
    """patches = [(layer, kind, pos), ...]; returns stage vectors at all
    positions we care about: (n_stages, N_ROWS, 2, 768) for pos 0 and GRAFT_P."""
    mod_hooks = [model.h[l].attn.register_forward_hook(patch_hook(kind, pos))
                 for l, kind, pos in patches]
    caps: dict[str, torch.Tensor] = {}
    cap_hooks = []
    for l, block in enumerate(model.h):
        cap_hooks.append(block.attn.register_forward_hook(
            lambda _m, _i, out, l=l: caps.__setitem__(f"attn{l}", out)))
        cap_hooks.append(block.register_forward_hook(
            lambda _m, _i, out, l=l: caps.__setitem__(f"block{l}", out)))
    cap_hooks.append(model.ln_f.register_forward_hook(
        lambda _m, _i, out: caps.__setitem__("ln_f", out)))

    keep = [0, GRAFT_P]
    vecs = np.zeros((len(stages), N_ROWS, len(keep), 768), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
            model(batch)
            hs = [model.wte(batch)]
            for l in range(n_layer):
                hs.append(hs[-1] + caps[f"attn{l}"])
                hs.append(caps[f"block{l}"])
            hs.append(caps["ln_f"])
            for si, h in enumerate(hs):
                vecs[si, start:start + len(batch)] = h[:, keep].float().cpu().numpy()
    for h in mod_hooks + cap_hooks:
        h.remove()
    return vecs


runs = {
    "baseline": run([]),
    "kill attn1 @0": run([(0, "kill", 0)]),
    "kill attn1+2 @0": run([(0, "kill", 0), (1, "kill", 0)]),
    "graft attn1 @256": run([(0, "graft", GRAFT_P)]),
    "graft attn1+2 @256": run([(0, "graft", GRAFT_P), (1, "graft", GRAFT_P)]),
}

# massive-vector direction: unit mean of baseline pos-0 after-MLP-2 unit vectors
si_mlp2 = stages.index("after MLP 2")
ok0 = tok0 != EOS_ID
okG = (tokG != EOS_ID) & (tokR != EOS_ID)
b0 = runs["baseline"][si_mlp2, ok0, 0]
u = (b0 / np.linalg.norm(b0, axis=1, keepdims=True)).mean(0)
u /= np.linalg.norm(u)

print(f"rows: {N_ROWS} ({ok0.sum()} used at pos 0, {okG.sum()} at pos {GRAFT_P})\n")
print(f"mean ||h|| by stage at the intervened position "
      f"(pos 0 for kill, pos {GRAFT_P} for graft):")
hdr = ["stage", "pos0 base", "kill a1", "kill a1+2", f"p{GRAFT_P} base",
       "graft a1", "graft a1+2"]
print("".join(s.ljust(19 if i == 0 else 12) for i, s in enumerate(hdr)))
norm_tbl = {}
for si, st in enumerate(stages):
    vals = [np.linalg.norm(runs["baseline"][si, ok0, 0], axis=1).mean(),
            np.linalg.norm(runs["kill attn1 @0"][si, ok0, 0], axis=1).mean(),
            np.linalg.norm(runs["kill attn1+2 @0"][si, ok0, 0], axis=1).mean(),
            np.linalg.norm(runs["baseline"][si, okG, 1], axis=1).mean(),
            np.linalg.norm(runs["graft attn1 @256"][si, okG, 1], axis=1).mean(),
            np.linalg.norm(runs["graft attn1+2 @256"][si, okG, 1], axis=1).mean()]
    norm_tbl[st] = vals
    print(st.ljust(19) + "".join(f"{v:12.2f}" for v in vals))

print(f"\nmean projection onto massive direction u at after-MLP-2:")
print(f"  pos 0:   baseline {(runs['baseline'][si_mlp2, ok0, 0] @ u).mean():8.1f}   "
      f"kill a1 {(runs['kill attn1 @0'][si_mlp2, ok0, 0] @ u).mean():8.1f}   "
      f"kill a1+2 {(runs['kill attn1+2 @0'][si_mlp2, ok0, 0] @ u).mean():8.1f}")
print(f"  pos {GRAFT_P}: baseline {(runs['baseline'][si_mlp2, okG, 1] @ u).mean():8.1f}   "
      f"graft a1 {(runs['graft attn1 @256'][si_mlp2, okG, 1] @ u).mean():8.1f}   "
      f"graft a1+2 {(runs['graft attn1+2 @256'][si_mlp2, okG, 1] @ u).mean():8.1f}")

# ---------------------------------------------------------------------- plot
fig, (axk, axg) = plt.subplots(1, 2, figsize=(13.5, 5))
xs = np.arange(len(stages))
short = [s.replace("after ", "").replace("attention", "attn") for s in stages]

for name, col, ls in [("baseline", "tab:orange", "-"),
                      ("kill attn1 @0", "tab:green", "--"),
                      ("kill attn1+2 @0", "tab:red", "-.")]:
    axk.plot(xs, [np.linalg.norm(runs[name][si, ok0, 0], axis=1).mean()
                  for si in range(len(stages))], ls, color=col, marker="o",
             ms=4, label=name)
axk.plot(xs, [np.linalg.norm(runs["baseline"][si, okG, 1], axis=1).mean()
              for si in range(len(stages))], ":", color="tab:blue", lw=1.5,
         label=f"bulk reference (pos {GRAFT_P}, baseline)")
axk.set_title("necessity: kill the attention output at position 0\n"
              "(replaced by the same row's output at position 300)", fontsize=11)

for name, col, ls in [("baseline", "tab:blue", "-"),
                      ("graft attn1 @256", "tab:green", "--"),
                      ("graft attn1+2 @256", "tab:red", "-.")]:
    axg.plot(xs, [np.linalg.norm(runs[name][si, okG, 1], axis=1).mean()
                  for si in range(len(stages))], ls, color=col, marker="o",
             ms=4, label=name)
axg.plot(xs, [np.linalg.norm(runs["baseline"][si, ok0, 0], axis=1).mean()
              for si in range(len(stages))], ":", color="tab:orange", lw=1.5,
         label="position-0 reference (baseline)")
axg.set_title(f"sufficiency: graft the self-only attention output\n"
              f"onto bulk position {GRAFT_P}", fontsize=11)

for ax in (axk, axg):
    ax.set_yscale("log")
    ax.set_xticks(xs, short, rotation=45, ha="right")
    ax.set_ylabel("mean ‖h‖ at the intervened position")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — causal patch: the massive vector follows the attention "
             f"position signal ({N_ROWS} Pile rows)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_patch.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {HERE.parent / 'pos0_patch.png'}")
