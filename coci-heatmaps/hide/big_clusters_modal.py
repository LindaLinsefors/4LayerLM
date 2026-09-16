"""Modal GPU job: fire-site statistics for all >=5-member alive co-CI>0.9
clusters of the two new decompositions (newA: 37 clusters, newB: 42 — from
cache/big_clusters.json, written by big_clusters_extract.py).

Per selected component: per-position CI>0.1 fire counts (512) and fires-on-EOS
count; per cluster: full fired-token histogram (bincount over the vocab,
weighted by how many members fire). Same 4,000 cached Pile rows.

Run:  modal run coci-heatmaps/hide/big_clusters_modal.py
Then: modal volume get vpd-4layer /big_clusters_newA.npz coci-heatmaps/hide/cache/
      modal volume get vpd-4layer /big_clusters_newB.npz coci-heatmaps/hide/cache/
Npz keys per site: "<site>|ids", "<site>|pos_counts" (n,512), "<site>|eos" (n,);
per cluster k of a site: "<site>#<k>|ids", "<site>#<k>|tok" (vocab histogram).
"""

import json
from pathlib import Path

import modal

app = modal.App("new-decomps-big-clusters")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "zstd")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
FIRE_THRESH = 0.1
VOCAB = 50277
EOS = 0


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600)
def compute(name: str, clusters: list[dict]) -> None:
    import time

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")

    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run

    run = RUNS[name]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / run, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys

    # per site: union of cluster member ids + local slices per cluster
    sites: dict[str, list] = {}
    for cl in clusters:
        sites.setdefault(cl["site"], []).append(cl["ids"])
    sel = {s: np.concatenate([np.array(c) for c in cls]) for s, cls in sites.items()}
    sel_j = {s: jnp.asarray(v) for s, v in sel.items()}
    print(f"{name}: {len(clusters)} clusters over {len(sites)} sites, "
          f"{sum(len(v) for v in sel.values())} comps", flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {s: jnp.clip(pre[s][..., sel_j[s]], 0.0, 1.0) > FIRE_THRESH
                for s in sites}

    rows = np.load("/data/pile_rows.npy")  # (4000, 512) int32
    B = 16
    pos_counts = {s: np.zeros((len(sel[s]), rows.shape[1]), np.int64) for s in sites}
    eos_counts = {s: np.zeros(len(sel[s]), np.int64) for s in sites}
    tok_counts = {s: [np.zeros(VOCAB, np.int64) for _ in cls]
                  for s, cls in sites.items()}
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for i in range(0, len(rows), B):
            tok = rows[i:i + B]
            out = step(placed, ci_fn, jnp.asarray(tok))
            for s, cls in sites.items():
                fires = np.asarray(out[s])  # (B, S, n_sel)
                pos_counts[s] += fires.sum(0).T
                eos_counts[s] += fires[tok == EOS].sum(0)
                off = 0
                for k, c in enumerate(cls):
                    w = fires[:, :, off:off + len(c)].sum(-1)
                    tok_counts[s][k] += np.bincount(
                        tok.ravel(), weights=w.ravel(), minlength=VOCAB
                    ).astype(np.int64)
                    off += len(c)
            if (i // B) % 50 == 0:
                print(f"batch {i // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    data = {}
    for s, cls in sites.items():
        data[f"{s}|ids"] = sel[s]
        data[f"{s}|pos_counts"] = pos_counts[s]
        data[f"{s}|eos"] = eos_counts[s]
        for k, c in enumerate(cls):
            data[f"{s}#{k}|ids"] = np.array(c)
            data[f"{s}#{k}|tok"] = tok_counts[s][k]
    np.savez_compressed(f"/data/big_clusters_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/big_clusters_{name}.npz ({time.time() - t0:.0f}s total)")


@app.local_entrypoint()
def main() -> None:
    path = Path(__file__).parent / "cache" / "big_clusters.json"
    all_clusters = json.loads(path.read_text())
    list(compute.starmap([(n, all_clusters[n]) for n in RUNS]))
