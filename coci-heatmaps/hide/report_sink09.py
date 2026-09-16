"""clustered09 ALIVE report for the sink-model decompositions (C/D/E).

Same threshold-rule clustering and heatmaps as report_new09.py (rules
2026-09-02: r > 0.9 chains, joins blocked at r <= 0), applied to the
100k-step JAX decompositions of the two attention-sink/untied-head targets:

  C = p-d60af588  (target t-87f91319, "sink seed 45", decomposition seed 0)
  D = p-fecd6a6b  (same target, seed 1)
  E = p-bd411e35  (target t-75f6c439, "sink seed 46", seed 0)

Data: cache/coci_<name>.npz from coci_compute_sink_modal.py (sample mean CI
+ co-CI r over the 4,000 cached Pile rows; alive = sample mean CI > 1e-6).
U/V factors come from compare-decomps/hide/cache/uv_<name>.npz
(sink_stats_modal.py) — unlike the newA/newB uv dumps those hold ALL
components in ascending id order, so no alive-id list is needed.

Unlike report_new09.py the cosine heatmaps are SIGNED (user request
2026-09-08): each component's (U, V) gauge is flipped jointly so its input
activation V_c.x is positive on the majority of CI>0.1 tokens (comp_signs
from interactive_cross.py; stats from compare-decomps/hide/cache/
act_signs_<name>.npz).

Writes <name>/report_alive_clustered09.md; figures in
hide/figures/<name>/alive_clustered09/.

Usage: python coci-heatmaps/hide/report_sink09.py [C D E]   (default C)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
UV_CACHE = HERE.parent.parent / "compare-decomps" / "hide" / "cache"
sys.path.insert(0, str(HERE))
from interactive_cross import comp_signs  # noqa: E402
from report import MATS  # noqa: E402
from report_clustered09 import R_BLOCK, R_EDGE, heatmap, rule_clusters  # noqa: E402

DECOMPS = {"C": ("p-d60af588", "t-87f91319", "sink seed 45", "seed 0"),
           "D": ("p-fecd6a6b", "t-87f91319", "sink seed 45", "seed 1"),
           "E": ("p-bd411e35", "t-75f6c439", "sink seed 46", "seed 0")}
N_LAYERS = 4


def main() -> None:
    only = sys.argv[1:] or ["C"]
    for name in only:
        run, target, tname, seed = DECOMPS[name]
        stats = np.load(CACHE / f"coci_{name}.npz")
        uv = np.load(UV_CACHE / f"uv_{name}.npz")
        acts = np.load(UV_CACHE / f"act_signs_{name}.npz")
        head = (
            f"Decomposition **{name} = `{run}`** ({seed}; 100k-step JAX "
            f"decomposition of the attention-sink/untied-head target "
            f"`{target}` = \"{tname}\", newA recipe: "
            "ImportanceMinimalityLoss frequency coeff 6.6e-5; see CLAUDE.md "
            "\"Attention-sink models & decompositions\"). Heatmaps of "
            "**Pearson r of per-token causal importance** (clip(preact, 0, "
            "1) = lower_leaky, continuous sampling) between every pair of a "
            "matrix's subcomponents, over the 4,000 cached Pile rows (2.05M "
            "tokens) — the same co-CI measure and threshold-rule clusters "
            "as the old decomposition's "
            "[report_alive_clustered09.md](../old/report_alive_clustered09.md): "
            f"**(1)** two components with co-CI r > {R_EDGE} are in the same "
            "cluster (pairs processed in descending r, chains allowed); "
            f"**(2)** a component does not join a cluster if it has r ≤ "
            f"{R_BLOCK} (or undefined r) with any existing member — such "
            "joins are skipped (count noted per matrix when > 0). Clusters "
            "are placed by descending mean member CI (singletons land where "
            "the plain mean-CI sort would put them), members within a "
            "cluster by descending mean CI. **Black outlines** mark "
            "multi-member clusters on the diagonal. **Gray** = zero CI "
            "variance in the sample (r undefined).\n\n"
            "No harvest DB exists for these runs, so mean CI is the "
            "**sample mean over the same 2.05M tokens** and alive is the "
            "proxy **sample mean CI > 1e-6**.\n\n"
            "Each matrix shows **three heatmaps in the same cluster "
            "order**: co-CI r, then the **signed cosine similarity between "
            "the components' write vectors cos(U_a, U_b)** (U = the output "
            "(d_out) factor of the rank-one subcomponent V_c U_c^T) and "
            "**between their read-in vectors cos(V_a, V_b)** (V = the "
            "input (d_in) factor). A component's sign is gauge ((V_c, U_c) "
            "-> (-V_c, -U_c) is the same component), so each component's "
            "(U_c, V_c) is flipped jointly to make its **input activation "
            "V_c·x positive on the majority of tokens where it fires (CI > "
            "0.1)** (ties by summed firing activation, never-firing "
            "components by the all-token activation sum; stats over the "
            "same 4,000 Pile rows). Same white-centered scale as co-CI r; "
            "random baseline E|cos| = sqrt(2/(pi d)) ~= 0.03 for d = 768, "
            "0.014 for d = 3072.")
        lines = [
            f"# {name} (`{run}`) — CI co-activation, alive components, "
            "co-CI>0.9 clusters",
            "",
            head,
            "",
        ]

        # thick rule between matrices, tripled between layers
        hr = '<hr style="height:10px;background:#555;border:none;">'
        for l in range(N_LAYERS):
            if l > 0:
                lines += [hr, ""] * 3
            lines += [f"### Layer {l}", ""]
            for i, mat in enumerate(MATS):
                if i > 0:
                    lines += [hr, ""]
                mod = f"h.{l}.{mat}"
                r = stats[f"{mod}|r"].astype(np.float32)
                m = stats[f"{mod}|mean"]
                order_all = np.argsort(-m, kind="stable")
                alive = order_all[m[order_all] > 1e-6]
                multi, blocked = rule_clusters(r, alive)
                in_multi = {int(c) for cl in multi for c in cl}
                clusters = ([c[np.argsort(-m[c], kind="stable")]
                             for c in multi]
                            + [np.array([c]) for c in alive
                               if int(c) not in in_multi])
                clusters.sort(key=lambda c: -m[c].mean())
                order = np.concatenate(clusters)
                sizes = [len(c) for c in clusters]
                fig_dir = HERE / "figures" / name / "alive_clustered09"
                fig_dir.mkdir(parents=True, exist_ok=True)
                fname = f"h{l}_{mat.replace('.', '_')}.png"
                n, nmulti = len(order), sum(s > 1 for s in sizes)
                title = (f"{name} ({run})  {mod}\nco-CI r(CI), {n} "
                         "alive components (co-CI>0.9 cluster order)")
                heatmap(r[np.ix_(order, order)], order, sizes, title,
                        fig_dir / fname)
                lines += [
                    f"#### {mod} — {n} components, {nmulti} clusters "
                    f"with ≥ 2 members (largest {max(sizes)})"
                    + (f", {blocked} joins blocked" if blocked else ""),
                    "",
                    f"![{mod}](../hide/figures/{name}/alive_clustered09/{fname})",
                    ""]
                # signed cos(U) / cos(V) in the same cluster order; uv dump
                # rows hold ALL components in ascending id order, gauge from
                # the majority-positive-activation convention
                signs = comp_signs(acts, mod)
                for meas in ("U", "V"):
                    M = uv[f"{mod}|{meas}"].astype(np.float32)
                    M = (signs[:, None] * M)[order]
                    nrm = np.linalg.norm(M, axis=1, keepdims=True)
                    nrm[nrm == 0] = np.inf
                    c = (M / nrm) @ (M / nrm).T
                    fname2 = f"h{l}_{mat.replace('.', '_')}_cos{meas}.png"
                    title2 = (f"{name} ({run})  {mod}\ncos({meas}) (signed, "
                              f"majority-positive-activation gauge), "
                              f"{n} alive components (same order)")
                    heatmap(c, order, sizes, title2, fig_dir / fname2)
                    lines += [
                        f"![{mod} cos{meas}](../hide/figures/{name}/"
                        f"alive_clustered09/{fname2})",
                        ""]
                print(f"{name} {mod}: n={n}, {nmulti} multi-clusters, "
                      f"largest {max(sizes)}, blocked {blocked}", flush=True)

        out_dir = HERE.parent / name
        out_dir.mkdir(exist_ok=True)
        path = out_dir / "report_alive_clustered09.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
