"""Are there pos-0 (chunk-start attention-sink) components in the sink-model
decompositions C/D/E — models whose base has a *built-in* learned sink slot?

Reads pos_fires_sink_{C,D,E}.npz (this folder's cache) and, for comparison, the
emergent-sink decompositions' pos_fires_{old,newA,newB}.npz from
coci-heatmaps/hide/cache/. Flag criterion = the interactive-cross-heatmap one:
total fires (CI > 0.1 over the 4,000 cached Pile rows) >= 20 AND > 50% of them
at position 0; also reported at a strict > 90% share.

Run:  python sink-models/hide/pos0_analysis.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CH = ROOT / "coci-heatmaps" / "hide" / "cache"

SOURCES = {
    "old":  CH / "pos_fires_old.npz",
    "newA": CH / "pos_fires_newA.npz",
    "newB": CH / "pos_fires_newB.npz",
    "C": HERE / "cache" / "pos_fires_sink_C.npz",
    "D": HERE / "cache" / "pos_fires_sink_D.npz",
    "E": HERE / "cache" / "pos_fires_sink_E.npz",
}
MIN_FIRES = 20


def main() -> None:
    per_mat = {}   # (mod, name) -> (n50, n90, ids50)
    totals = {}
    mods = None
    for name, path in SOURCES.items():
        z = np.load(path)
        mods = [k[:-3] for k in z.files if k.endswith("|Fp")]
        mods.sort(key=lambda m: (int(m.split(".")[1]), m.split(".", 2)[2]))
        n50 = n90 = 0
        for m in mods:
            fp = z[f"{m}|Fp"]                      # (C, 512)
            tot = fp.sum(1)
            share = np.divide(fp[:, 0], tot, out=np.zeros(len(fp)),
                              where=tot > 0)
            sel = tot >= MIN_FIRES
            ids50 = np.nonzero(sel & (share > 0.5))[0]
            ids90 = np.nonzero(sel & (share > 0.9))[0]
            per_mat[(m, name)] = (len(ids50), len(ids90), ids50)
            n50 += len(ids50)
            n90 += len(ids90)
        totals[name] = (n50, n90)

    names = list(SOURCES)
    print(f"pos-0 components (>= {MIN_FIRES} fires, pos-0 fire share > 50% / > 90%)")
    print(f"{'matrix':<18}" + "".join(f"{n:>12}" for n in names))
    for m in mods:
        cells = []
        for n in names:
            c50, c90, _ = per_mat.get((m, n), (0, 0, None))
            cells.append(f"{c50}/{c90}")
        print(f"{m:<18}" + "".join(f"{c:>12}" for c in cells))
    print(f"{'TOTAL':<18}" + "".join(
        f"{totals[n][0]}/{totals[n][1]}".rjust(12) for n in names))

    # list the sink-model pos-0 comps with their fire pattern
    for name in ["C", "D", "E"]:
        z = np.load(SOURCES[name])
        print(f"\n--- {name}: pos-0 components (share > 50%) ---")
        for m in mods:
            _, _, ids = per_mat[(m, name)]
            for i in ids:
                fp = z[f"{m}|Fp"][i]
                tot = int(fp.sum())
                print(f"  {m}:{i}  fires {tot}, pos0 {fp[0]} "
                      f"({fp[0] / tot:.0%}), pos1 {fp[1]}, pos>=2 {fp[2:].sum()}")


if __name__ == "__main__":
    main()
