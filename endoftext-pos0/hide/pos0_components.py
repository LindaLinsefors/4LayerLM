"""Which VPD subcomponents implement the position-0 / EOS massive-vector write?

Mechanical attribution using the rank-one structure. For a decomposed Linear
with input x, subcomponent c contributes exactly s_c U_c to the output, where
s_c = x . V[:,c]. So the write of MLP 2's down_proj onto the massive direction
u decomposes exactly as  y.u = sum_c s_c (U_c . u),  and each component's mean
share at position-0 / EOS / bulk samples can be tabulated. Same logic
attributes the earlier stages onto their mean-difference directions:

  * h.0.attn.o_proj  -> direction of the attn-1 output mean difference (pos0 - bulk)
    (v/q/k pre-activations at layer 0 are position-blind — their input is the
    embedding stream — so o_proj components are the FIRST whose activations
    can carry position; position enters between v_proj and o_proj, in the
    softmax mixture.)
  * h.0.mlp.down_proj -> direction of the after-MLP-1 residual mean difference
  * h.1.attn.o_proj  -> direction of the attn-2 output mean difference
  * h.1.mlp.c_fc      -> activation contrast |a_c| at pos0 vs bulk (feeds the
    gelu, nonlinear, so ranked by contrast and verified by ablation)
  * h.1.mlp.down_proj -> projection onto u (the headline attribution)

Verification: surgical ablation — subtract the top-k components' rank-one
matrices outer(U_c, V_c) from the actual model weight, rerun, measure the
massive vector at position 0 / EOS / bulk (plus a random-k control).

Data: first N_ROWS cached Pile rows (bulk sample = position 300).
Output: endoftext-pos0/pos0_components.png + printed tables.
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
N_ROWS_ABL = 200
BATCH_SIZE = 16
EOS_ID = 0
T = 512
BULK_P = 300
MODULES = [f"h.{l}.{m}" for l in (0, 1)
           for m in ("attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]

rows = torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt",
                  map_location="cpu", weights_only=True)[:N_ROWS]
rows = [r[:T] for r in rows]

model, pc, _ = load_pile_4l()
model = model.to(DEVICE)
V = {m: pc.components[m].V.to(DEVICE) for m in MODULES}      # (d_in, C)
U = {m: pc.components[m].U.to(DEVICE) for m in MODULES}      # (C, d_out)


def get_module(name):
    obj = model
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


# ------------------------------------------------------- pass 1: collect
# per module: component pre-activations s_c at pos-0 / EOS / bulk samples;
# plus attn outputs and block outputs for the directions.
caps: dict[str, torch.Tensor] = {}
hooks = []
for m in MODULES:
    hooks.append(get_module(m).register_forward_pre_hook(
        lambda _m, inp, m=m: caps.__setitem__(m, inp[0])))
for l in (0, 1):
    hooks.append(model.h[l].attn.register_forward_hook(
        lambda _m, _i, out, l=l: caps.__setitem__(f"attnout{l}", out)))
    hooks.append(model.h[l].register_forward_hook(
        lambda _m, _i, out, l=l: caps.__setitem__(f"blockout{l}", out)))

S = {m: {g: [] for g in ("pos0", "eos", "bulk")} for m in MODULES}
D = {k: {g: [] for g in ("pos0", "eos", "bulk")}
     for k in ("attnout0", "attnout1", "blockout0", "blockout1")}
with torch.no_grad():
    for start in range(0, N_ROWS, BATCH_SIZE):
        batch = torch.stack(rows[start:start + BATCH_SIZE]).to(DEVICE)
        model(batch)
        tok = batch.cpu().numpy()
        sel = {"pos0": [(bi, 0) for bi in range(len(tok)) if tok[bi, 0] != EOS_ID],
               "eos": [(bi, p) for bi, p in zip(*np.nonzero(tok == EOS_ID)) if p > 0],
               "bulk": [(bi, BULK_P) for bi in range(len(tok))
                        if tok[bi, BULK_P] != EOS_ID]}
        for g, idx in sel.items():
            if not idx:
                continue
            b_i = torch.tensor([i[0] for i in idx], device=DEVICE)
            p_i = torch.tensor([i[1] for i in idx], device=DEVICE)
            for m in MODULES:
                S[m][g].append((caps[m][b_i, p_i] @ V[m]).float().cpu())
            for k in D:
                D[k][g].append(caps[k][b_i, p_i].float().cpu())
for h in hooks:
    h.remove()
S = {m: {g: torch.cat(v).double().numpy() for g, v in gg.items()} for m, gg in S.items()}
D = {k: {g: torch.cat(v).double().numpy() for g, v in gg.items()} for k, gg in D.items()}
n = {g: len(S[MODULES[0]][g]) for g in ("pos0", "eos", "bulk")}
print(f"samples: pos0 {n['pos0']}, EOS {n['eos']}, bulk {n['bulk']}")

# directions
def unit(v):
    return v / np.linalg.norm(v)

u = unit(D["blockout1"]["pos0"].mean(0))                       # massive direction
d_a1 = unit(D["attnout0"]["pos0"].mean(0) - D["attnout0"]["bulk"].mean(0))
d_m1 = unit(D["blockout0"]["pos0"].mean(0) - D["blockout0"]["bulk"].mean(0))
d_a2 = unit(D["attnout1"]["pos0"].mean(0) - D["attnout1"]["bulk"].mean(0))

Un = {m: U[m].double().cpu().numpy() for m in MODULES}

def contrib(m, direction):
    """per-component mean projection s_c (U_c . dir) for each group -> (C, 3)."""
    Ud = Un[m] @ direction                                     # (C,)
    return {g: S[m][g].mean(0) * Ud for g in ("pos0", "eos", "bulk")}, Ud


def top_table(name, vals, Ud, k=12, extra=None):
    order = np.argsort(-np.abs(vals["pos0"]))[:k]
    total = {g: vals[g].sum() for g in vals}
    print(f"\n{name}: total projection pos0 {total['pos0']:+.1f}, "
          f"EOS {total['eos']:+.1f}, bulk {total['bulk']:+.1f}")
    print("  comp      pos0       EOS      bulk    |U_c.dir|")
    for c in order:
        print(f"  {c:5d} {vals['pos0'][c]:+9.2f} {vals['eos'][c]:+9.2f} "
              f"{vals['bulk'][c]:+9.2f}   {abs(Ud[c]):7.3f}")
    return order


# headline: down_proj of MLP 2 onto u
vals_dp, Ud_dp = contrib("h.1.mlp.down_proj", u)
order_dp = np.argsort(-np.abs(vals_dp["pos0"]))
top_table("h.1.mlp.down_proj write onto massive direction u", vals_dp, Ud_dp, k=15)
csum = np.cumsum(vals_dp["pos0"][order_dp]) / vals_dp["pos0"].sum()
for k in (1, 2, 3, 5, 10, 20, 50):
    print(f"  top-{k}: {csum[k - 1]:.1%} of the u-write")

# c_fc of MLP 2: activation contrast
m = "h.1.mlp.c_fc"
a = {g: np.abs(S[m][g]).mean(0) * np.linalg.norm(Un[m], axis=1) for g in S[m]}
diff = a["pos0"] - a["bulk"]
order_cfc = np.argsort(-diff)
print(f"\nh.1.mlp.c_fc: top-12 by activation contrast E|a_c| pos0 - bulk:")
print("  comp   |a| pos0   |a| EOS   |a| bulk   ratio p0/bulk")
for c in order_cfc[:12]:
    print(f"  {c:5d} {a['pos0'][c]:9.2f} {a['eos'][c]:9.2f} {a['bulk'][c]:9.2f}"
          f"   {a['pos0'][c] / max(a['bulk'][c], 1e-9):9.1f}")

# earlier stages
vals_o0, Ud_o0 = contrib("h.0.attn.o_proj", d_a1)
top_table("h.0.attn.o_proj onto attn-1 mean-diff direction", vals_o0, Ud_o0)
vals_m1, Ud_m1 = contrib("h.0.mlp.down_proj", d_m1)
top_table("h.0.mlp.down_proj onto after-MLP-1 mean-diff direction", vals_m1, Ud_m1)
vals_o1, Ud_o1 = contrib("h.1.attn.o_proj", d_a2)
top_table("h.1.attn.o_proj onto attn-2 mean-diff direction", vals_o1, Ud_o1)

# ------------------------------------------------------- pass 2: ablation
rows_abl = rows[:N_ROWS_ABL]
tok0_ok = np.array([int(r[0]) != EOS_ID for r in rows_abl])
tokB_ok = np.array([int(r[BULK_P]) != EOS_ID for r in rows_abl])
eos_idx = [(ri, p) for ri, r in enumerate(rows_abl)
           for p in np.nonzero(r.numpy() == EOS_ID)[0] if p > 0]


def measure():
    """mean ||h|| after MLP 2 and mean proj on u at pos0 / EOS / bulk."""
    cap = {}
    hk = model.h[1].register_forward_hook(lambda _m, _i, out: cap.__setitem__("b1", out))
    outs = {"pos0": [], "eos": [], "bulk": []}
    with torch.no_grad():
        for start in range(0, N_ROWS_ABL, BATCH_SIZE):
            batch = torch.stack(rows_abl[start:start + BATCH_SIZE]).to(DEVICE)
            model(batch)
            b1 = cap["b1"].float().cpu().numpy()
            outs["pos0"].append(b1[:, 0])
            outs["bulk"].append(b1[:, BULK_P])
            for ri, p in eos_idx:
                if start <= ri < start + len(batch):
                    outs["eos"].append(b1[ri - start, p][None])
    hk.remove()
    res = {}
    for g, sel in (("pos0", tok0_ok), ("bulk", tokB_ok)):
        v = np.concatenate([o for o in outs[g]])[: N_ROWS_ABL][sel]
        res[g] = (np.linalg.norm(v, axis=1).mean(), (v @ u).mean())
    v = np.concatenate(outs["eos"]) if outs["eos"] else np.zeros((0, 768))
    res["eos"] = (np.linalg.norm(v, axis=1).mean(), (v @ u).mean())
    return res


def ablate(module: str, comps: np.ndarray):
    """subtract sum_c outer(U_c, V_c) from the module's weight; return restore fn."""
    lin = get_module(module)
    delta = (V[module][:, comps] @ U[module][comps, :]).T.to(lin.weight.dtype)
    lin.weight.data -= delta
    return lambda: lin.weight.data.add_(delta)


print("\nablation: mean ||h|| after MLP 2 (and proj on u) at pos0 / EOS / bulk")
base = measure()
print(f"  baseline           : pos0 {base['pos0'][0]:6.1f} ({base['pos0'][1]:6.1f})  "
      f"EOS {base['eos'][0]:6.1f} ({base['eos'][1]:6.1f})  "
      f"bulk {base['bulk'][0]:5.1f} ({base['bulk'][1]:5.2f})")
abl_results = {}
for k in (1, 2, 3, 5, 10, 20, 50):
    restore = ablate("h.1.mlp.down_proj", order_dp[:k])
    r = measure()
    restore()
    abl_results[f"dp{k}"] = r
    print(f"  down_proj top-{k:3d} : pos0 {r['pos0'][0]:6.1f} ({r['pos0'][1]:6.1f})  "
          f"EOS {r['eos'][0]:6.1f} ({r['eos'][1]:6.1f})  "
      f"bulk {r['bulk'][0]:5.1f} ({r['bulk'][1]:5.2f})")
rng = np.random.default_rng(0)
restore = ablate("h.1.mlp.down_proj", rng.choice(Un["h.1.mlp.down_proj"].shape[0],
                                                 20, replace=False))
r = measure()
restore()
print(f"  down_proj rand-20  : pos0 {r['pos0'][0]:6.1f} ({r['pos0'][1]:6.1f})  "
      f"EOS {r['eos'][0]:6.1f} ({r['eos'][1]:6.1f})  "
      f"bulk {r['bulk'][0]:5.1f} ({r['bulk'][1]:5.2f})")
for k in (5, 10, 20, 50):
    restore = ablate("h.1.mlp.c_fc", order_cfc[:k])
    r = measure()
    restore()
    abl_results[f"cfc{k}"] = r
    print(f"  c_fc top-{k:3d}      : pos0 {r['pos0'][0]:6.1f} ({r['pos0'][1]:6.1f})  "
          f"EOS {r['eos'][0]:6.1f} ({r['eos'][1]:6.1f})  "
          f"bulk {r['bulk'][0]:5.1f} ({r['bulk'][1]:5.2f})")

# ---------------------------------------------------------------------- plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax_sort, ax_cum), (ax_cfc, ax_abl) = axes

o = order_dp
ax_sort.semilogy(np.abs(vals_dp["pos0"][o]), lw=1.2, label="|mean proj| at pos 0")
ax_sort.semilogy(np.abs(vals_dp["eos"][o]), lw=1.2, alpha=0.7, label="at EOS")
ax_sort.semilogy(np.abs(vals_dp["bulk"][o]), lw=1.2, alpha=0.7, label="at bulk")
ax_sort.set_xlabel("h.1.mlp.down_proj components, sorted by |pos-0 contribution|")
ax_sort.set_ylabel("|mean s_c (U_c·u)|")
ax_sort.set_title("per-component write onto the massive direction u", fontsize=11)
ax_sort.legend(fontsize=9, frameon=False)

ax_cum.plot(np.arange(1, len(csum) + 1), csum, lw=1.5)
ax_cum.set_xscale("log")
ax_cum.axhline(1.0, color="k", ls=":", lw=0.8)
ax_cum.set_xlabel("top-k components (by |pos-0 contribution|)")
ax_cum.set_ylabel("cumulative share of the u-write at pos 0")
ax_cum.set_title("how concentrated is the massive write?", fontsize=11)

ax_cfc.loglog(np.maximum(a["bulk"], 1e-4), np.maximum(a["pos0"], 1e-4), ".",
              ms=3, alpha=0.4)
ax_cfc.loglog(np.maximum(a["bulk"][order_cfc[:20]], 1e-4),
              np.maximum(a["pos0"][order_cfc[:20]], 1e-4), "r.", ms=6,
              label="top-20 by contrast")
lims = [1e-3, max(a["pos0"].max(), a["bulk"].max()) * 1.5]
ax_cfc.plot(lims, lims, "k:", lw=0.8)
ax_cfc.set_xlabel("E|a_c| at bulk")
ax_cfc.set_ylabel("E|a_c| at pos 0")
ax_cfc.set_title("h.1.mlp.c_fc: activation contrast pos 0 vs bulk", fontsize=11)
ax_cfc.legend(fontsize=9, frameon=False)

ks = [1, 2, 3, 5, 10, 20, 50]
ax_abl.plot(ks, [abl_results[f"dp{k}"]["pos0"][0] for k in ks], "o-",
            label="ablate down_proj top-k, pos 0")
ax_abl.plot(ks, [abl_results[f"dp{k}"]["eos"][0] for k in ks], "s-",
            label="… EOS")
ks2 = [5, 10, 20, 50]
ax_abl.plot(ks2, [abl_results[f"cfc{k}"]["pos0"][0] for k in ks2], "o--",
            label="ablate c_fc top-k, pos 0")
ax_abl.plot(ks2, [abl_results[f"cfc{k}"]["eos"][0] for k in ks2], "s--",
            label="… EOS")
ax_abl.axhline(base["pos0"][0], color="tab:orange", ls=":", lw=1,
               label=f"baseline pos 0 ({base['pos0'][0]:.0f})")
ax_abl.axhline(base["bulk"][0], color="k", ls=":", lw=1,
               label=f"baseline bulk ({base['bulk'][0]:.1f})")
ax_abl.set_xscale("log")
ax_abl.set_xlabel("k components ablated")
ax_abl.set_ylabel("mean ‖h‖ after MLP 2")
ax_abl.set_title("surgical ablation of h.1.mlp components", fontsize=11)
ax_abl.legend(fontsize=8, frameon=False)

for ax in axes.flat:
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"pile_4l — which subcomponents write the massive vector "
             f"({N_ROWS} rows attribution, {N_ROWS_ABL} rows ablation)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE.parent / "pos0_components.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {HERE.parent / 'pos0_components.png'}")

np.savez(HERE / "cache" / "pos0_components.npz",
         u=u, d_a1=d_a1, d_m1=d_m1, d_a2=d_a2,
         **{f"{m}|{g}": vals[g] for m, (vals, _) in
            [("h.1.mlp.down_proj", (vals_dp, Ud_dp)),
             ("h.0.attn.o_proj", (vals_o0, Ud_o0)),
             ("h.0.mlp.down_proj", (vals_m1, Ud_m1)),
             ("h.1.attn.o_proj", (vals_o1, Ud_o1))] for g in vals},
         **{f"h.1.mlp.c_fc|a_{g}": a[g] for g in a})
print(f"saved {HERE / 'cache' / 'pos0_components.npz'}")
