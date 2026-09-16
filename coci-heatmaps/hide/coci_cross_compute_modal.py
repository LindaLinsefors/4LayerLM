"""Modal GPU job: CROSS-decomposition co-CI sufficient statistics between the
old decomposition (s-55ea3f9b, torch) and the two new 800k-step JAX
decompositions newA = p-8383f5e5 / newB = p-4d9a6a12, per matrix, for the
pairs (old, newA), (old, newB), (newA, newB).

Alive components only (old: harvest mean CI > 1e-6; newA/newB: sample mean
CI > 1e-6), index sets in /cross_alive.npz on the volume (built + uploaded by
the local snippet that wrote cache/cross_alive.npz). All three decompositions
are evaluated on the SAME 2.05M tokens (4,000 cached Pile rows x 512, batches
of 16 rows = 8,192 tokens, identical batch order), so per-token CI series can
be crossed batch by batch:

  stage old  (torch image): per batch, dump the concatenated alive-component
      CI (8192, 9973) float16 to /data/cross_ci/old_b<i>.npy  (~41 GB).
  stage newA (JAX image): compute newA CI per batch, accumulate float64
      S1/S2 for old + newA and the cross-Gram sum(ci_old ci_A) per matrix;
      also dump newA's alive CI (~69 GB) -> /data/cross_partial_newA.npz.
  stage newB (JAX image): same vs old AND vs the newA dumps ->
      /data/cross_partial_newB.npz  (S1/S2 newB, Grams old x B and A x B).

Pearson r is assembled locally by report_cross.py from the partials
(r = (G/T - mx myT) / (sx syT)). Old/newA series entering the Grams are the
float16 dumps (quantization ~1e-3 relative - irrelevant for heatmaps).

Run:  modal run coci-heatmaps/hide/coci_cross_compute_modal.py
      (stages run sequentially; --stage old|newA|newB|cleanup for one stage)
Then: modal volume get vpd-4layer /cross_partial_newA.npz coci-heatmaps/hide/cache/
      modal volume get vpd-4layer /cross_partial_newB.npz coci-heatmaps/hide/cache/
After verifying: modal run ... --stage cleanup   (deletes the ~110 GB of dumps)
"""

import modal

app = modal.App("coci-cross")
vol = modal.Volume.from_name("vpd-4layer")

torch_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)
COMMIT = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
jax_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

RUNS = {"newA": "p-8383f5e5", "newB": "p-4d9a6a12"}
MODS = [f"h.{l}.{m}" for l in range(4)
        for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                  "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]
B = 16          # rows per batch -> 250 batches of 8,192 tokens
DUMP = "/data/cross_ci"


def _alive(which: str):
    import numpy as np
    z = np.load("/data/cross_alive.npz")
    return {m: z[f"{which}|{m}"] for m in MODS}


@app.function(image=torch_image, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def dump_old() -> None:
    """Dump the old decomposition's alive-component per-token CI, per batch."""
    import os
    import time
    from pathlib import Path

    os.environ["PARAM_DECOMP_OUT_DIR"] = "/data/param_decomp_out"

    import numpy as np
    import torch

    import param_decomp.pretrain.run_info as pri

    def _local_files(entity, project, run_id):
        d = Path(f"/data/param_decomp_out/pretrain_cache/{project}-{run_id}")
        assert d.exists(), d
        return pri.WandbDownloadedFiles(
            checkpoint=d / "model_step_99999.pt", config=d / "final_config.yaml",
            model_config=d / "model_config.yaml", tokenizer=d / "tokenizer.json")

    pri._download_wandb_files = _local_files
    from param_decomp.models.component_model import ComponentModel

    torch.backends.cuda.matmul.allow_tf32 = False
    alive = _alive("old")
    Path(DUMP).mkdir(exist_ok=True)

    model = ComponentModel.from_pretrained("/data/vpd/model_400000.pth")
    model.eval().to("cuda")
    rows = torch.stack([r[:512] for r in torch.load("/data/pile_rows.pt")])
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(rows), B):
            tok = rows[b:b + B].to("cuda")
            out = model(tok, cache_type="input")
            ci = model.calc_causal_importances(out.cache, sampling="continuous")
            parts = [ci.lower_leaky[m].reshape(tok.numel(), -1)
                     [:, torch.as_tensor(alive[m], device="cuda")].half()
                     for m in MODS]
            np.save(f"{DUMP}/old_b{b // B:03d}.npy",
                    torch.cat(parts, dim=1).cpu().numpy())
            if (b // B) % 25 == 0:
                print(f"batch {b // B + 1}/{len(rows) // B}, "
                      f"{time.time() - t0:.0f}s", flush=True)
                vol.commit()
    vol.commit()
    print(f"dumped old CI ({time.time() - t0:.0f}s)")


@app.function(image=jax_image, gpu="A10G", volumes={"/data": vol},
              timeout=7200, memory=32768)
def new_pass(name: str) -> None:
    """newA: old x A cross-Grams + dump A's CI.  newB: old x B and A x B."""
    import time
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from jax.sharding import PartitionSpec as P

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run

    run = RUNS[name]
    root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    loaded = open_jax_run(root / "runs" / run, step=800000, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    assert set(MODS) == set(ci_fn.fn.output_names)

    alive_self = _alive(name)
    others = ["old"] if name == "newA" else ["old", "newA"]
    alive_oth = {o: _alive(o) for o in others}
    # column offsets of each matrix inside a concatenated dump file
    off = {o: np.concatenate([[0], np.cumsum([len(alive_oth[o][m])
                                              for m in MODS])])
           for o in others}

    @eqx.filter_jit
    def step(placed, ci_fn, tokens, oth_parts):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        s1, s2, grams, dump = {}, {}, {}, {}
        for m in MODS:
            c = jnp.clip(pre[m], 0.0, 1.0).reshape(-1, pre[m].shape[-1])
            c = c[:, alive_self[m]]
            s1[m], s2[m] = c.sum(0), (c * c).sum(0)
            grams[m] = [jnp.einsum("to,tc->oc", p[m], c,
                                   out_sharding=P(None, None))
                        for p in oth_parts]
            dump[m] = c.astype(jnp.float16)
        return s1, s2, grams, dump

    nself = {m: len(alive_self[m]) for m in MODS}
    S1 = {m: np.zeros(nself[m]) for m in MODS}
    S2 = {m: np.zeros(nself[m]) for m in MODS}
    G = {o: {m: np.zeros((len(alive_oth[o][m]), nself[m])) for m in MODS}
         for o in others}
    S1o = {m: np.zeros(len(alive_oth["old"][m])) for m in MODS}
    S2o = {m: np.zeros_like(S1o[m]) for m in MODS}

    rows = np.load("/data/pile_rows.npy")
    nb = len(rows) // B
    T = 0
    t0 = time.time()
    with jax.set_mesh(loaded.mesh):
        for b in range(nb):
            full = {o: np.load(f"{DUMP}/{o}_b{b:03d}.npy").astype(np.float32)
                    for o in others}
            parts = [{m: full[o][:, off[o][i]:off[o][i + 1]]
                      for i, m in enumerate(MODS)} for o in others]
            tok = jnp.asarray(rows[b * B:(b + 1) * B])
            s1, s2, grams, dump = step(placed, ci_fn, tok, parts)
            for m in MODS:
                S1[m] += np.asarray(s1[m], np.float64)
                S2[m] += np.asarray(s2[m], np.float64)
                for o, g in zip(others, grams[m]):
                    G[o][m] += np.asarray(g, np.float64)
                if name == "newA":  # old sums, from the same f16 series
                    x = parts[0][m].astype(np.float64)
                    S1o[m] += x.sum(0)
                    S2o[m] += (x * x).sum(0)
            if name == "newA":
                np.save(f"{DUMP}/newA_b{b:03d}.npy",
                        np.concatenate([np.asarray(dump[m]) for m in MODS],
                                       axis=1))
            T += tok.size
            if b % 25 == 0:
                print(f"batch {b + 1}/{nb}, {T} tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)
                if name == "newA":
                    vol.commit()

    data = {"T": np.array(T)}
    for m in MODS:
        data[f"{m}|S1_{name}"] = S1[m]
        data[f"{m}|S2_{name}"] = S2[m]
        for o in others:
            data[f"{m}|G_{o}_{name}"] = G[o][m].astype(np.float32)
        if name == "newA":
            data[f"{m}|S1_old"] = S1o[m]
            data[f"{m}|S2_old"] = S2o[m]
    np.savez(f"/data/cross_partial_{name}.npz", **data)
    vol.commit()
    print(f"saved /data/cross_partial_{name}.npz ({time.time() - t0:.0f}s total)")


@app.function(image=jax_image, volumes={"/data": vol}, timeout=3600)
def cleanup() -> None:
    import shutil
    shutil.rmtree(DUMP, ignore_errors=True)
    vol.commit()
    print("removed", DUMP)


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    if stage in ("all", "old"):
        dump_old.remote()
    if stage in ("all", "newA"):
        new_pass.remote("newA")
    if stage in ("all", "newB"):
        new_pass.remote("newB")
    if stage == "cleanup":
        cleanup.remote()
