"""RoPE-wavelength spectra of the pile_4l q/k subcomponents.

For each decomposed attention matrix h.<l>.attn.{q,k}_proj, each subcomponent c
writes the rank-one output U[c,:] (V[:,c] . x) into q/k space (n_heads x 128 dims,
head-major).  RoPE rotates dim pairs (p, p+64) within each head; the rotation-
invariant energy a component puts into wavelength plane p is

    E[c,p] = sum_h ( U[c, h*128+p]^2 + U[c, h*128+p+64]^2 ),   p = 0..63

with wavelength lambda_p = 2*pi * 10000^(p/64) tokens.  The heatmaps show the
per-component *fraction* F[c,p] = E[c,p] / sum_p E[c,p] (independent of overall
component magnitude; uniform = 1/64).

Outputs (one level up): qk_wavelengths.png (per-component heatmaps, all 8
matrices) and qk_wavelengths_mean.png (mean alive-component spectrum per layer).
Cache: cache/energies.npz.  Run: python spectra.py  (~2 min first time, then fast).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent            # pile-qk-comps/hide
ROOT = HERE.parents[1]
OUT = HERE.parent                       # pile-qk-comps/
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

N_PLANES = 64
BASE = 10000
LAMBDA = 2 * np.pi * BASE ** (np.arange(N_PLANES) / N_PLANES)   # tokens
N_CTX = 512
MATRICES = [f"h.{l}.attn.{m}" for l in range(4) for m in ("q_proj", "k_proj")]
ALIVE, SEMI, NEVER = 0, 1, 2            # category codes from component-analasys
CAT_NAMES = ["alive", "semi", "never"]

ENERGY_NPZ = CACHE / "energies.npz"
MEAN_CI_NPZ = CACHE / "mean_ci.npz"
MAGNITUDES_NPZ = ROOT / "component-analasys" / "hide" / "cache" / "magnitudes.npz"
HARVEST_DB = (ROOT / "prev_paper" / "models" / "pile_4layer"
              / "additional-component-data" / "harvest.db")


def compute() -> dict[str, np.ndarray]:
    """Per matrix: E[c,h,p] (head x plane energies, key "<mod>|hp") plus the
    two marginals E[c,p] (key "<mod>") and E[c,h] (key "<mod>|head"), cached."""
    if ENERGY_NPZ.exists():
        cached = dict(np.load(ENERGY_NPZ))
        if any(k.endswith("|V") for k in cached):
            return cached
    sys.path.insert(0, str(ROOT))
    from load import PILE_4L, ParameterComponents

    (ckpt,) = PILE_4L.glob("vpd_decomposition_*/model_400000.pth")
    comps = ParameterComponents(ckpt).components
    energies = {}
    for mod in MATRICES:
        U = comps[mod].U.double().numpy()               # (C, n_heads*128)
        Uh = U.reshape(U.shape[0], -1, 128)             # (C, n_heads, 128)
        hp = Uh[:, :, :64] ** 2 + Uh[:, :, 64:] ** 2    # (C, n_heads, 64)
        energies[mod + "|hp"] = hp
        energies[mod] = hp.sum(axis=1)
        energies[mod + "|head"] = hp.sum(axis=2)
        energies[mod + "|V"] = comps[mod].V.double().numpy().T   # (C, d_in) read-in vectors
    np.savez_compressed(ENERGY_NPZ, **energies)
    return energies


def mean_ci() -> dict[str, np.ndarray]:
    """Mean causal importance per component (0 for never-fired), from harvest.db."""
    if MEAN_CI_NPZ.exists():
        return dict(np.load(MEAN_CI_NPZ))
    import json
    import sqlite3

    out = {mod: np.zeros(512) for mod in MATRICES}
    con = sqlite3.connect(f"file:{HARVEST_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT layer, component_idx, mean_activations FROM components "
        f"WHERE layer IN ({','.join('?' * len(MATRICES))})", MATRICES)
    for layer, idx, ma in rows:
        out[layer][idx] = json.loads(ma)["causal_importance"]
    con.close()
    np.savez(MEAN_CI_NPZ, **out)
    return out


def categories(mod: str, C: int) -> np.ndarray:
    z = np.load(MAGNITUDES_NPZ, allow_pickle=True)
    mask = z["module"] == mod
    cat = np.full(C, -1)
    cat[z["index"][mask]] = z["category"][mask]
    assert (cat >= 0).all()
    return cat


# ---------------------------------------------------------------- plotting ----

TICK_PLANES = [0, 8, 16, 24, 32, 40, 48, 56, 63]
TICK_LABELS = ["6.3", "20", "63", "200", "630", "2.0k", "6.3k", "20k", "54k"]
CTX_PLANE = N_PLANES * np.log(N_CTX / (2 * np.pi)) / np.log(BASE)   # ~30.6
LAYER_RAMP = ["#7fb0e3", "#4a8fd0", "#2262ac", "#123c6e"]           # ordinal, layer 0->3


def plot_heatmaps(energies: dict[str, np.ndarray]) -> None:
    # alive components only; each figure row sized by its layer's larger count
    alive_F = {}
    for mod in MATRICES:
        E = energies[mod]
        F = E / E.sum(axis=1, keepdims=True)
        F = F[categories(mod, len(F)) == ALIVE]
        alive_F[mod] = F[np.argsort(F @ np.arange(N_PLANES))]   # sort by spectral CoM
    ratios = [max(len(alive_F[f"h.{l}.attn.q_proj"]),
                  len(alive_F[f"h.{l}.attn.k_proj"])) for l in range(4)]
    fig, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True,
                             gridspec_kw={"height_ratios": [max(r, 40) for r in ratios]})
    for ax, mod in zip(axes.ravel(), MATRICES):
        F = alive_F[mod]
        im = ax.imshow(F, aspect="auto", interpolation="antialiased",
                       cmap="Blues", vmin=0, vmax=0.06)
        ax.axvline(CTX_PLANE, color="#8a8a8a", lw=1, ls="--")
        ax.set_title(f"{mod}   ({len(F)} alive of C = 512)", fontsize=11)
        ax.set_xticks(TICK_PLANES, TICK_LABELS)
        ax.set_xlabel("wavelength  $\\lambda_p = 2\\pi\\cdot10000^{p/64}$  [tokens]",
                      fontsize=9)
        ax.set_ylabel("alive subcomponent\n(sorted by spectral CoM)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, shrink=0.4, pad=0.06,
                 label="fraction of output energy in plane,  $F_{c,p}$ (uniform = 1/64)")
    fig.suptitle("pile_4l — alive-subcomponent output energy across RoPE wavelength planes\n"
                 "dashed line: $\\lambda = n_{ctx} = 512$", fontsize=13)
    fig.savefig(OUT / "qk_wavelengths.png", dpi=200)
    plt.close(fig)


def plot_heads(energies: dict[str, np.ndarray]) -> None:
    """Same figure as plot_heatmaps but x = head instead of wavelength plane:
    E[c,h] = sum_d U[c, 128h+d]^2, row-normalized (uniform = 1/6).
    Rows sorted by dominant head, then by descending dominance."""
    alive_F = {}
    for mod in MATRICES:
        E = energies[mod + "|head"]                              # (C, 6)
        F = E / E.sum(axis=1, keepdims=True)
        F = F[categories(mod, len(F)) == ALIVE]
        alive_F[mod] = F[np.lexsort((-F.max(axis=1), F.argmax(axis=1)))]
    ratios = [max(len(alive_F[f"h.{l}.attn.q_proj"]),
                  len(alive_F[f"h.{l}.attn.k_proj"])) for l in range(4)]
    fig, axes = plt.subplots(4, 2, figsize=(9, 14), constrained_layout=True,
                             gridspec_kw={"height_ratios": [max(r, 40) for r in ratios]})
    for ax, mod in zip(axes.ravel(), MATRICES):
        F = alive_F[mod]
        im = ax.imshow(F, aspect="auto", interpolation="antialiased",
                       cmap="Blues", vmin=0, vmax=1)
        for b in np.arange(0.5, 5.5):                            # head separators
            ax.axvline(b, color="white", lw=1.5)
        ax.set_title(f"{mod} — {len(F)} alive", fontsize=10)
        ax.set_xticks(range(6))
        ax.set_xlabel("head", fontsize=9)
        ax.set_ylabel("alive subcomponent\n(sorted by dominant head)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, shrink=0.4, pad=0.06,
                 label="fraction of output energy in head,  $E_{c,h}/\\sum_h E_{c,h}$ (uniform = 1/6)")
    fig.suptitle("pile_4l — alive-subcomponent output energy per attention head\n"
                 "$E_{c,h} = \\sum_d U_{c,\\,128h+d}^2$, row-normalized", fontsize=13)
    fig.savefig(OUT / "qk_heads.png", dpi=200)
    plt.close(fig)


def plot_heads_wavelengths(energies: dict[str, np.ndarray]) -> None:
    """The full 3-way view: per component, per head, per wavelength plane.
    x = 6 head blocks x 64 planes (384 columns), row-normalized (uniform =
    1/384).  Rows sorted by dominant head, then by spectral CoM."""
    alive_F = {}
    for mod in MATRICES:
        hp = energies[mod + "|hp"]                               # (C, 6, 64)
        hp = hp[categories(mod, len(hp)) == ALIVE]
        F = hp / hp.sum(axis=(1, 2), keepdims=True)
        dom = F.sum(axis=2).argmax(axis=1)
        com = F.sum(axis=1) @ np.arange(N_PLANES)
        alive_F[mod] = F[np.lexsort((com, dom))].reshape(len(F), -1)
    ratios = [max(len(alive_F[f"h.{l}.attn.q_proj"]),
                  len(alive_F[f"h.{l}.attn.k_proj"])) for l in range(4)]
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True,
                             gridspec_kw={"height_ratios": [max(r, 40) for r in ratios]})
    for ax, mod in zip(axes.ravel(), MATRICES):
        F = alive_F[mod]
        im = ax.imshow(F, aspect="auto", interpolation="antialiased",
                       cmap="Blues", vmin=0, vmax=0.01)
        for h in range(6):
            if h:                                                # head separators
                ax.axvline(h * N_PLANES - 0.5, color="#d94f4f", lw=1)
            ax.axvline(h * N_PLANES + CTX_PLANE, color="#8a8a8a", lw=0.8, ls="--")
        ax.set_title(f"{mod} — {len(F)} alive", fontsize=10)
        ax.set_xticks([h * N_PLANES + t for h in range(6) for t in (0, 32)],
                      ["6.3", "630"] * 6, fontsize=6)
        secax = ax.secondary_xaxis("top")
        secax.set_xticks([h * N_PLANES + 31.5 for h in range(6)],
                         [f"head {h}" for h in range(6)], fontsize=8)
        secax.tick_params(length=0)
        ax.set_xlabel("wavelength [tokens] within each head block", fontsize=8)
        ax.set_ylabel("alive subcomponent\n(dominant head, then CoM)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, shrink=0.4, pad=0.03,
                 label="fraction of output energy in (head, plane) cell (uniform = 1/384)")
    fig.suptitle("pile_4l — alive-subcomponent output energy per head and RoPE wavelength plane\n"
                 "$E_{c,h,p} = U_{c,\\,128h+p}^2 + U_{c,\\,128h+p+64}^2$, row-normalized;  "
                 "red lines: head boundaries;  dashed: $\\lambda = n_{ctx} = 512$", fontsize=13)
    fig.savefig(OUT / "qk_heads_wavelengths.png", dpi=200)
    plt.close(fig)


def plot_readin_cosine(energies: dict[str, np.ndarray], sort: str = "cluster") -> None:
    """|cosine| between the read-in vectors V_c of all alive q/k components,
    all 8 matrices in one matrix (blocks in MATRICES order).  Within a block,
    sort = "cluster": hierarchical clustering so similar read-in directions sit
    together (average linkage on distance 1 - |cos|, optimal leaf ordering);
    sort = "ci": descending mean causal importance (from harvest.db).
    |.| because the sign of V_c is gauge: (V_c, U_c) -> (-V_c, -U_c) leaves
    the rank-one component unchanged."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    vecs, labels = [], []
    for mod in MATRICES:
        alive = categories(mod, len(energies[mod])) == ALIVE
        V = energies[mod + "|V"][alive]
        Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
        if sort == "cluster":
            D = squareform(np.clip(1 - np.abs(Vn @ Vn.T), 0, None), checks=False)
            order = leaves_list(optimal_leaf_ordering(linkage(D, method="average"), D))
        else:
            order = np.argsort(-mean_ci()[mod][alive])
        vecs.append(Vn[order])
        labels.append(f"L{mod.split('.')[1]} {mod.split('.')[3][0]}")
    M = np.concatenate(vecs)
    S = np.abs(M @ M.T)
    bounds = np.cumsum([len(v) for v in vecs])
    # pixel-exact rendering: 3 screen px per matrix cell, so rows never blur
    n, ppc, dpi = len(S), 3, 100
    ax_in = n * ppc / dpi
    lm, rm, bm, tm = 0.9, 1.6, 0.5, 1.0                      # margins [inches]
    fig = plt.figure(figsize=(lm + ax_in + rm, bm + ax_in + tm), dpi=dpi)
    ax = fig.add_axes([lm / (lm + ax_in + rm), bm / (bm + ax_in + tm),
                       ax_in / (lm + ax_in + rm), ax_in / (bm + ax_in + tm)])
    im = ax.imshow(S, cmap="Blues", vmin=0, vmax=0.3, interpolation="nearest")
    for i, b in enumerate(bounds[:-1]):
        color = "#d94f4f" if i % 2 == 0 else "#222222"       # q|k divide red, layer divide black
        ax.axhline(b - 0.5, color=color, lw=0.8)
        ax.axvline(b - 0.5, color=color, lw=0.8)
    centers = bounds - np.diff(np.concatenate(([0], bounds))) / 2
    ax.set_xticks(centers, labels, fontsize=9)
    ax.set_yticks(centers, labels, fontsize=9)
    ax.tick_params(length=0)
    cax = fig.add_axes([(lm + ax_in + 0.45) / (lm + ax_in + rm),
                        bm / (bm + ax_in + tm),
                        0.25 / (lm + ax_in + rm),
                        0.8 * ax_in / (bm + ax_in + tm)])
    fig.colorbar(im, cax=cax,
                 label="$|\\cos(V_a, V_b)|$  (color capped at 0.3; random $\\approx 1/\\sqrt{768} = 0.036$)")
    sort_desc = ("hierarchically clustered (average linkage on $1-|\\cos|$)"
                 if sort == "cluster" else "sorted by descending mean causal importance")
    ax.set_title("pile_4l — |cosine| between read-in vectors $V_c$ of all alive q/k "
                 "subcomponents\nblack lines: layer divides, red lines: q|k divides; "
                 f"within a block, {sort_desc}", fontsize=13)
    suffix = "" if sort == "cluster" else "_ci"
    fig.savefig(OUT / f"qk_readin_cosine{suffix}.png", dpi=dpi)
    plt.close(fig)


def plot_means(energies: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for ax, kind in zip(axes, ("q_proj", "k_proj")):
        for l in range(4):
            mod = f"h.{l}.attn.{kind}"
            E = energies[mod]
            F = E / E.sum(axis=1, keepdims=True)
            alive = categories(mod, len(F)) == ALIVE
            ax.plot(np.arange(N_PLANES), F[alive].mean(axis=0),
                    color=LAYER_RAMP[l], lw=2, label=f"layer {l}")
        ax.axhline(1 / N_PLANES, color="#999999", lw=1, ls=":", zorder=0)
        ax.text(0.5, 1 / N_PLANES, "uniform 1/64", fontsize=8, color="#777777",
                va="bottom")
        ax.axvline(CTX_PLANE, color="#8a8a8a", lw=1, ls="--")
        ax.text(CTX_PLANE + 0.7, ax.get_ylim()[1], "$n_{ctx}$", fontsize=8,
                color="#8a8a8a", va="top")
        ax.set_title(kind, fontsize=11)
        ax.set_xticks(TICK_PLANES, TICK_LABELS)
        ax.set_xlabel("wavelength [tokens]", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("mean energy fraction over alive components", fontsize=9)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("pile_4l — mean RoPE-wavelength spectrum of alive subcomponents", fontsize=12)
    fig.savefig(OUT / "qk_wavelengths_mean.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    energies = compute()
    for mod in MATRICES:
        E = energies[mod]
        F = E / E.sum(axis=1, keepdims=True)
        print(f"{mod}: C={len(E)}, max fraction {F.max():.3f}, "
              f"median row-max {np.median(F.max(axis=1)):.3f}")
    plot_heatmaps(energies)
    plot_heads(energies)
    plot_heads_wavelengths(energies)
    plot_readin_cosine(energies)
    plot_readin_cosine(energies, sort="ci")
    plot_means(energies)
    print("wrote qk_wavelengths.png, qk_heads.png, qk_heads_wavelengths.png,"
          " qk_wavelengths_mean.png in", OUT)
