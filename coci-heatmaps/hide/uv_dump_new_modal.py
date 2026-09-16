"""Modal job: dump the alive components' U/V factors of the two NEW 800k-step
JAX decompositions (newA = p-8383f5e5, newB = p-4d9a6a12) as npz.

For every matrix, `loaded.prepared_weights[kind]` holds the layer-stacked
factors with V (n_layer, d_in, C) and U (n_layer, C, d_out) — the same
orientation as the torch checkpoints (site write = ((x @ V) * mask) @ U, see
glu_transformer.decomposed_site_output). Alive component ids come from
/data/cross_alive.npz (upload coci-heatmaps/hide/cache/cross_alive.npz first;
alive = sample mean CI > 1e-6, the same selection as all cross reports).

Setup (decomposition archives in volume vpd-4layer) is the one made by
coci_compute_new_modal.py.

Run:  modal volume put vpd-4layer coci-heatmaps/hide/cache/cross_alive.npz /cross_alive.npz
      modal run coci-heatmaps/hide/uv_dump_new_modal.py
Then: modal volume get vpd-4layer /uv_newA.npz coci-heatmaps/hide/cache/
      modal volume get vpd-4layer /uv_newB.npz coci-heatmaps/hide/cache/
Output npz keys per module: "<mod>|U" (n_alive, d_out) float16,
"<mod>|V" (n_alive, d_in) float16 (V columns as rows), rows in ascending
alive-component-id order (= the cross_alive order).
"""

import modal

app = modal.App("uv-dump-new-decomps")
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
DIMS = {"attn.q_proj": (768, 768), "attn.k_proj": (768, 768),
        "attn.v_proj": (768, 768), "attn.o_proj": (768, 768),
        "mlp.c_fc": (768, 3072), "mlp.down_proj": (3072, 768)}  # (d_in, d_out)


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=1800)
def dump(name: str) -> None:
    from pathlib import Path

    import numpy as np

    from param_decomp.experiments.lm.load_run import open_jax_run

    run = RUNS[name]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / run, step=800000, data_root=root)
    pw = loaded.prepared_weights
    sites = list(loaded.ci_fn.fn.output_names)
    print(f"{name} = {run}: kinds {sorted(pw)}", flush=True)

    alive = np.load("/data/cross_alive.npz")
    data = {}
    for site in sites:  # "h.<l>.<attn|mlp>.<kind>"
        _, l, mod = site.split(".", 2)
        layer = int(l)
        kind = mod.split(".")[-1]  # pw kinds are short: q_proj, c_fc, ...
        assert kind in pw, f"{kind} not in {sorted(pw)}"
        V = np.asarray(pw[kind]["V"][layer], np.float32)  # (d_in, C)
        U = np.asarray(pw[kind]["U"][layer], np.float32)  # (C, d_out)
        d_in, d_out = DIMS[mod]
        assert V.shape[0] == d_in and U.shape[1] == d_out and V.shape[1] == U.shape[0], \
            f"{site}: V {V.shape}, U {U.shape}"
        ids = alive[f"{name}|{site}"]
        data[f"{site}|V"] = V[:, ids].T.astype(np.float16)
        data[f"{site}|U"] = U[ids].astype(np.float16)
        print(f"{site}: {len(ids)} alive, V {data[f'{site}|V'].shape}, "
              f"U {data[f'{site}|U'].shape}", flush=True)
    np.savez(f"/data/uv_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/uv_{name}.npz")


@app.local_entrypoint()
def main() -> None:
    list(dump.map(["newA", "newB"]))
