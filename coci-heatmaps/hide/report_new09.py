"""clustered09 reports for the two NEW 800k-step decompositions (newA/newB).

Same threshold-rule clustering and heatmaps as report_clustered09.py (rules
2026-09-02: r > 0.9 chains, joins blocked at r <= 0), applied to the new JAX
decompositions of the pile_4l target:

  newA = p-8383f5e5  (ImportanceMinimalityLoss frequency coeff 6.6e-5)
  newB = p-4d9a6a12  (coeff 6.6e-6)

Differences from the old-decomposition reports: there is no harvest DB for the
new runs, so mean CI is the SAMPLE mean over the same 2.05M tokens the co-CI
was computed on (cache/coci_<name>.npz from coci_compute_new_modal.py), and
"alive" is the proxy sample mean CI > 1e-6.

The ALIVE report additionally shows, per matrix and in the same cluster
order, |cos(U)| and |cos(V)| heatmaps between the components' write (U,
d_out) and read-in (V, d_in) factors (cache/uv_<name>.npz from
uv_dump_new_modal.py; |.| because component sign is gauge).

Writes newA/report_{alive,all}_clustered09.md (and newB/...); figures in
hide/figures/<name>/<alive|all>_clustered09/.

Usage: python coci-heatmaps/hide/report_new09.py   (~6 min, mostly rasterizing)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
sys.path.insert(0, str(HERE))
from report import MATS  # noqa: E402
from report_clustered09 import R_BLOCK, R_EDGE, heatmap, rule_clusters  # noqa: E402

DECOMPS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
COEFF = {"newA": "6.6e-5", "newB": "6.6e-6"}
N_LAYERS = 4


def main() -> None:
    only = sys.argv[1:] or list(DECOMPS)
    cross_alive = np.load(CACHE / "cross_alive.npz")
    for name in only:
        run = DECOMPS[name]
        stats = np.load(CACHE / f"coci_{name}.npz")
        uv = np.load(CACHE / f"uv_{name}.npz")
        head = (
            f"Decomposition **{name} = `{run}`** (800k-step JAX decomposition "
            "of the pile_4l target, ImportanceMinimalityLoss frequency coeff "
            f"{COEFF[name]}; see CLAUDE.md \"New 800k-step decompositions\"). "
            "Heatmaps of **Pearson r of per-token causal importance** "
            "(clip(preact, 0, 1) = lower_leaky, continuous sampling) between "
            "every pair of a matrix's subcomponents, over the 4,000 cached "
            "Pile rows (2.05M tokens) — the same co-CI measure and "
            "threshold-rule clusters as the old decomposition's "
            "[report_{v}_clustered09.md](../old/report_{v}_clustered09.md): "
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
            "No harvest DB exists for the new runs, so mean CI is the "
            "**sample mean over the same 2.05M tokens** and alive is the "
            "proxy **sample mean CI > 1e-6**{cut}{cos}")
        cut_note = (
            "; the **dashed green line** sits at that alive count in plain "
            "mean-CI order (cluster order can put a few components on the "
            "wrong side of it).")
        cos_note = (
            "\n\nEach matrix shows **three heatmaps in the same cluster "
            "order**: co-CI r, then the **absolute cosine similarity between "
            "the components' write vectors |cos(U_a, U_b)|** (U = the output "
            "(d_out) factor of the rank-one subcomponent V_c U_c^T) and "
            "**between their read-in vectors |cos(V_a, V_b)|** (V = the "
            "input (d_in) factor). Absolute value because a component's sign "
            "is gauge ((V_c, U_c) -> (-V_c, -U_c) is the same component); "
            "Reds, 0 -> 1; random baseline E|cos| = sqrt(2/(pi d)) ~= 0.03 "
            "for d = 768, 0.014 for d = 3072.")
        lines = {v: [
            f"# {name} (`{run}`) — CI co-activation, {v} components, "
            "co-CI>0.9 clusters",
            "",
            head.replace("{v}", v)
                .replace("{cut}", cut_note if v == "all" else ".")
                .replace("{cos}", cos_note if v == "alive" else ""),
            "",
        ] for v in ("alive", "all")}

        for l in range(N_LAYERS):
            for v in lines:
                lines[v] += [f"### Layer {l}", ""]
            for mat in MATS:
                mod = f"h.{l}.{mat}"
                r = stats[f"{mod}|r"].astype(np.float32)
                m = stats[f"{mod}|mean"]
                order_all = np.argsort(-m, kind="stable")
                alive = order_all[m[order_all] > 1e-6]
                for v, sel in (("alive", alive), ("all", order_all)):
                    multi, blocked = rule_clusters(r, sel)
                    in_multi = {int(c) for cl in multi for c in cl}
                    clusters = ([c[np.argsort(-m[c], kind="stable")]
                                 for c in multi]
                                + [np.array([c]) for c in sel
                                   if int(c) not in in_multi])
                    clusters.sort(key=lambda c: -m[c].mean())
                    order = np.concatenate(clusters)
                    sizes = [len(c) for c in clusters]
                    fig_dir = HERE / "figures" / name / f"{v}_clustered09"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"h{l}_{mat.replace('.', '_')}.png"
                    n, nmulti = len(order), sum(s > 1 for s in sizes)
                    title = (f"{name} ({run})  {mod}\nco-CI r(CI), {n} "
                             + ("alive " if v == "alive" else "")
                             + "components (co-CI>0.9 cluster order)")
                    heatmap(r[np.ix_(order, order)], order, sizes, title,
                            fig_dir / fname,
                            cutoff=len(alive) if v == "all" else None)
                    lines[v] += [
                        f"#### {mod} — {n} components, {nmulti} clusters "
                        f"with ≥ 2 members (largest {max(sizes)})"
                        + (f", {blocked} joins blocked" if blocked else ""),
                        "",
                        f"![{mod}](../hide/figures/{name}/{v}_clustered09/{fname})",
                        ""]
                    if v == "alive":
                        # |cos(U)| / |cos(V)| in the same cluster order; the
                        # uv dump rows are in ascending alive-id order
                        a = cross_alive[f"{name}|{mod}"]
                        assert set(order.tolist()) == set(a.tolist())
                        pos = np.searchsorted(a, order)
                        for meas in ("U", "V"):
                            M = uv[f"{mod}|{meas}"].astype(np.float32)[pos]
                            nrm = np.linalg.norm(M, axis=1, keepdims=True)
                            nrm[nrm == 0] = np.inf
                            c = np.abs((M / nrm) @ (M / nrm).T)
                            fname2 = (f"h{l}_{mat.replace('.', '_')}"
                                      f"_cos{meas}.png")
                            title2 = (f"{name} ({run})  {mod}\n|cos({meas})|, "
                                      f"{n} alive components (same order)")
                            heatmap(c, order, sizes, title2, fig_dir / fname2,
                                    cmap="Reds", vmin=0, vmax=1)
                            lines[v] += [
                                f"![{mod} cos{meas}](../hide/figures/{name}/"
                                f"{v}_clustered09/{fname2})",
                                ""]
                    print(f"{name} {mod} [{v}]: n={n}, {nmulti} multi-clusters, "
                          f"largest {max(sizes)}, blocked {blocked}", flush=True)

        out_dir = HERE.parent / name
        out_dir.mkdir(exist_ok=True)
        for v in ("alive", "all"):
            path = out_dir / f"report_{v}_clustered09.md"
            path.write_text("\n".join(lines[v]), encoding="utf-8")
            print("wrote", path)


if __name__ == "__main__":
    main()
