"""Per-matrix component-group reports: report.md + pairwise grid plots.

For each module in groups_def.GROUPS, creates pile-qk-comps/L<l>-Attn-<q|k>/
with report.md listing every group (members, autointerp labels, mean CI,
CI-firing count in the sample) and, per group with >1 member, one PNG with
three annotated pairwise grids:
  co-activation  : Pearson r of |a_c| over the 1.02M-token sample
  co-CI          : Pearson r of causal importance over the same sample
  input cosine   : |cos(V_a, V_b)| of the read-in vectors
Stats from cache/groups_stats.npz (see groups_compute.py); V from energies.npz.

Usage: python groups_report.py [module ...]   (default: all in GROUPS)
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
OUT_BASE = HERE.parent
sys.path.insert(0, str(HERE))
from groups_def import GROUPS

STATS = np.load(HERE / "cache" / "groups_stats.npz")
ENERGIES = np.load(HERE / "cache" / "energies.npz")
MEAN_CI = np.load(HERE / "cache" / "mean_ci.npz")
T = int(STATS["T"])
INTERP_DB = (ROOT / "prev_paper" / "models" / "pile_4layer"
             / "additional-component-data" / "interp.db")


def folder_name(mod: str) -> str:
    l, kind = mod.split(".")[1], mod.split(".")[3][0]
    return f"L{l}-Attn-{kind}"


def pearson_from_stats(S1: np.ndarray, G: np.ndarray) -> np.ndarray:
    m = S1 / T
    cov = G / T - np.outer(m, m)
    s = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.outer(s, s)
    return r


def grid_panel(ax, M, ids, title, cmap, vmin, vmax):
    disp = np.ma.masked_invalid(M)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    im = ax.imshow(disp, cmap=cm, vmin=vmin, vmax=vmax)
    n = len(ids)
    fs = max(5, min(9, 90 // n))
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                dark = abs(M[i, j]) > 0.6 * max(abs(vmin), abs(vmax))
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=fs, color="white" if dark else "#222222")
    ax.set_xticks(range(n), ids, rotation=90, fontsize=fs + 1)
    ax.set_yticks(range(n), ids, fontsize=fs + 1)
    ax.set_title(title, fontsize=10)
    return im


def make_module(mod: str) -> None:
    out_dir = OUT_BASE / folder_name(mod)
    out_dir.mkdir(exist_ok=True)
    idx = STATS[f"{mod}|idx"]
    pos = {int(c): k for k, c in enumerate(idx)}
    r_act = pearson_from_stats(STATS[f"{mod}|S1a"], STATS[f"{mod}|Ga"])
    r_ci = pearson_from_stats(STATS[f"{mod}|S1c"], STATS[f"{mod}|Gc"])
    fires = STATS[f"{mod}|F"]
    V = ENERGIES[f"{mod}|V"]
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos = np.abs(Vn @ Vn.T)
    mci = MEAN_CI[mod]

    con = sqlite3.connect(f"file:{INTERP_DB}?mode=ro", uri=True)

    def label(c: int) -> tuple[str, str]:
        r = con.execute("SELECT label, confidence FROM interpretations "
                        "WHERE component_key=?", (f"{mod}:{c}",)).fetchone()
        return (r[0], r[1]) if r else ("(no label)", "-")

    groups = GROUPS[mod]
    lines = [
        f"# {folder_name(mod)} — component groups by function ({mod})",
        "",
        f"All {len(idx)} alive components (harvest mean CI > 1e-6), grouped by "
        "autointerp label similarity (grouping by Claude, 2026-08-28; labels from "
        "the local interp.db — the paper website used a different autointerp run, "
        "so its label wording differs).",
        "",
        "Per group with >1 member, three pairwise grids over a 2000-row "
        f"(= {T:,} token) Pile sample:",
        "",
        "- **co-activation** — Pearson r of |a_c| = |‖U_c‖ (V_c·φ)| per token",
        "- **co-CI** — Pearson r of the causal importance (lower_leaky) per token;"
        " gray = component never varied (no CI in sample)",
        "- **input cosine** — |cos(V_a, V_b)| of read-in vectors"
        " (|·|: component sign is gauge)",
        "",
        "'fires' below = tokens with CI > 0.1 in the sample.",
        "",
        "## Groups",
        "",
        "| group | n | members |",
        "|---|---|---|",
    ]
    for slug, title, members in groups:
        lines.append(f"| [{title}](#{slug}) | {len(members)} | "
                     + " ".join(str(c) for c in members) + " |")
    lines.append("")

    for slug, title, members in groups:
        lines += [f'<a id="{slug}"></a>', "", f"### {title} ({len(members)})", ""]
        for c in members:
            lab, conf = label(c)
            k = pos[c]
            lines.append(f"- **{c}** ({conf}, mean CI {mci[c]:.4f}, "
                         f"fires {int(fires[k])}): {lab}")
        lines.append("")
        if len(members) < 2:
            continue
        sel = [pos[c] for c in members]
        fig_w = max(9, 0.55 * len(members) * 3 + 4)
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_w / 3 + 0.8),
                                 constrained_layout=True)
        grid_panel(axes[0], r_act[np.ix_(sel, sel)], members,
                   "co-activation  r(|a|)", "RdBu_r", -1, 1)
        grid_panel(axes[1], r_ci[np.ix_(sel, sel)], members,
                   "co-CI  r(CI)", "RdBu_r", -1, 1)
        grid_panel(axes[2], cos[np.ix_(sel, sel)], members,
                   "input |cos(V)|", "Blues", 0, 1)
        fig.suptitle(f"{mod} — {title}", fontsize=11)
        fig.savefig(out_dir / f"{slug}.png", dpi=150)
        plt.close(fig)
        lines += [f"![{title}]({slug}.png)", ""]

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_dir / "report.md",
          f"({sum(1 for _, _, m in groups if len(m) > 1)} figures)")


if __name__ == "__main__":
    mods = sys.argv[1:] or list(GROUPS)
    for mod in mods:
        make_module(mod)
