"""Modal GPU job: cross-decomposition co-CI r for the 12 NEW pairs among
{old, newA, newB, C, D, E} (the 3 pile-trio pairs already exist in
coci-heatmaps/hide/cache/cross_partial_{newA,newB}.npz).

Design (faster than the chained coci_cross_compute_modal.py): all six
decompositions' alive-component per-token CI series are dumped IN PARALLEL
(6 GPUs, same 4,000 Pile rows in identical 16-row batches), then ONE plain
torch-CUDA stage streams all six dumps batch-aligned — reading each dump
exactly once — accumulating per-decomposition S1/S2 and the 12 pair Grams in
float64 on GPU, and writes the finished Pearson r (float16) per pair per
matrix.

Alive sets: old/newA/newB from /cross_alive.npz (as before); C/D/E = sample
mean CI > 1e-6 from /coci_{C,D,E}.npz (sink_stats_modal.py must have run).
Dumps land in /data/cross_ci/<name>_b<i>.npy (f16, ~450 GB total — delete
with --stage cleanup after verifying).

Run:  modal run compare-decomps/hide/cross_all_modal.py --stage dumps-trio
      modal run compare-decomps/hide/cross_all_modal.py --stage dumps-sink
      modal run compare-decomps/hide/cross_all_modal.py --stage grams
Then: modal volume get vpd-4layer /cross_r_new12.npz compare-decomps/hide/cache/
After verifying: modal run ... --stage cleanup
Output npz keys: "<a>|<b>|<mod>|r" (n_alive_a x n_alive_b, float16; a before b
in old,newA,newB,C,D,E order), plus "<name>|<mod>|S1"/"S2" (float64) and "T".

F extension (2026-09-19, decomposition F = p-c45e0001, the corrected-RoPE
re-decomposition of C's target t-87f91319): stage `dump-f` dumps F's alive CI
(fitted-RoPE forward — F's training-time forward; needs /coci_F.npz from
sink_stats_modal.py first), and stage `grams-f` streams all SEVEN dumps
(re-run dumps-trio + dumps-sink first if /data/cross_ci was cleaned) to write
the 6 F-pairs -> /cross_r_F.npz (keys "<a>|F|<mod>|r" + S1/S2 + "T").
"""

import modal

app = modal.App("compare-decomps-cross")
vol = modal.Volume.from_name("vpd-4layer")

torch_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("git+https://github.com/goodfire-ai/param-decomp@vpd-paper")
    .pip_install("wandb==0.23.1", "wandb-workspaces==0.1.12")
)
PINNED = "facf2e7b1d5273985ea1af277240f340f5bdeac0"
SINK = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"


def _jax_image(commit):
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("git", "curl", "zstd")
            .pip_install("param-decomp[cuda] @ "
                         f"git+https://github.com/goodfire-ai/param-decomp.git@{commit}"))


jax_pinned = _jax_image(PINNED)
jax_sink = _jax_image(SINK)
gram_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "numpy")

DEC = ["old", "newA", "newB", "C", "D", "E"]
NEW_RUNS = {"newA": ("p-8383f5e5", 800000), "newB": ("p-4d9a6a12", 800000)}
SINK_RUNS = {"C": ("p-d60af588", 100000), "D": ("p-fecd6a6b", 100000),
             "E": ("p-bd411e35", 100000), "F": ("p-c45e0001", 100000)}
FITTED_ROPE = {"F"}  # trained with the fitted corrected RoPE spectrum
F_PAIRS = [(a, "F") for a in DEC]
TRIO = {"old", "newA", "newB"}
NEW_PAIRS = [(a, b) for i, a in enumerate(DEC) for b in DEC[i + 1:]
             if not (a in TRIO and b in TRIO)]
MODS = [f"h.{l}.{m}" for l in range(4)
        for m in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                  "attn.o_proj", "mlp.c_fc", "mlp.down_proj")]
B = 16
NB = 250
DUMP = "/data/cross_ci"


def _alive(which: str):
    import numpy as np
    if which in TRIO:
        z = np.load("/data/cross_alive.npz")
        return {m: z[f"{which}|{m}"] for m in MODS}
    z = np.load(f"/data/coci_{which}.npz")
    return {m: np.flatnonzero(z[f"{m}|mean"] > 1e-6) for m in MODS}


@app.function(image=torch_image, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def dump_old() -> None:
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


def _dump_jax(name: str) -> None:
    """Dump one JAX decomposition's alive-component CI, per batch (shared by
    the pinned-commit newA/newB and sink-commit C/D/E stages)."""
    import time
    from pathlib import Path

    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.core.ci_fn import ci_preactivations
    from param_decomp.core.model import select_captures
    from param_decomp.experiments.lm.load_run import open_jax_run

    if name in NEW_RUNS:
        run, step_n = NEW_RUNS[name]
        root = Path(f"/data/new-decomps/{run}-decomposition-800000")
    else:
        run, step_n = SINK_RUNS[name]
        root = Path("/data/sink-models")
    if name in FITTED_ROPE:
        # F was trained with this spectrum monkeypatched in — same patch here
        from param_decomp.targets import llama_simple_mlp as lsm
        lif = np.load("/data/fitted_freqs_avg.npz")["log_inv_freq"]
        spec = jnp.asarray(np.exp(np.asarray(lif, np.float64)), jnp.float32)
        lsm.plain_rope_inv_freq = lambda cfg: spec
    loaded = open_jax_run(root / "runs" / run, step=step_n, data_root=root)
    placed, ci_fn = loaded.placed, loaded.ci_fn
    keys = ci_fn.fn.capture_keys
    assert set(MODS) == set(ci_fn.fn.output_names)
    if name in FITTED_ROPE:
        # sanity: fitted spectrum ⇒ target NLL ~2.5; broken base-1e4 gives ~8
        r8 = np.load("/data/pile_rows.npy")[:8]
        with jax.set_mesh(loaded.mesh):
            logits = np.asarray(placed.clean_forward(
                jnp.asarray(r8), keys).output, np.float32)
        mx = logits[:, :-1].max(-1, keepdims=True)
        lse = np.log(np.exp(logits[:, :-1] - mx).sum(-1)) + mx[..., 0]
        nll = float((lse - np.take_along_axis(
            logits[:, :-1], r8[:, 1:, None], -1)[..., 0]).mean())
        print(f"{name} target NLL (fitted RoPE): {nll:.3f}", flush=True)
        assert nll < 4.0, f"RoPE patch did not take effect (NLL {nll:.2f})"
    alive = _alive(name)
    Path(DUMP).mkdir(exist_ok=True)
    print(f"{name} = {run}: {sum(len(v) for v in alive.values())} alive comps",
          flush=True)

    @eqx.filter_jit
    def step(placed, ci_fn, tokens):
        caps = placed.clean_forward(tokens, keys).captures
        pre = ci_preactivations(ci_fn, select_captures(caps, keys), remat=False)
        return {m: jnp.clip(pre[m], 0.0, 1.0).reshape(-1, pre[m].shape[-1])
                [:, alive[m]].astype(jnp.float16) for m in MODS}

    rows = np.load("/data/pile_rows.npy")
    t0 = time.time()
    for b in range(NB):
        with jax.set_mesh(loaded.mesh):
            out = step(placed, ci_fn, jnp.asarray(rows[b * B:(b + 1) * B]))
        np.save(f"{DUMP}/{name}_b{b:03d}.npy",
                np.concatenate([np.asarray(out[m]) for m in MODS], axis=1))
        if b % 25 == 0:
            print(f"batch {b + 1}/{NB}, {time.time() - t0:.0f}s", flush=True)
            vol.commit()
    vol.commit()
    print(f"dumped {name} CI ({time.time() - t0:.0f}s)")


@app.function(image=jax_pinned, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def dump_new(name: str) -> None:
    _dump_jax(name)


@app.function(image=jax_sink, gpu="A10G", volumes={"/data": vol}, timeout=7200)
def dump_sink(name: str) -> None:
    _dump_jax(name)


@app.function(image=gram_image, gpu="A10G", volumes={"/data": vol},
              timeout=7200, memory=65536)
def grams() -> None:
    """Stream all six dumps batch-aligned once; accumulate S1/S2 + the 12 new
    pair Grams in float64 on GPU; write finished Pearson r (f16)."""
    import time

    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    alive = {n: _alive(n) for n in DEC}
    n_of = {n: {m: len(alive[n][m]) for m in MODS} for n in DEC}
    off = {n: np.concatenate([[0], np.cumsum([n_of[n][m] for m in MODS])])
           for n in DEC}

    S1 = {n: {m: torch.zeros(n_of[n][m], dtype=torch.float64, device="cuda")
              for m in MODS} for n in DEC}
    S2 = {n: {m: torch.zeros_like(S1[n][m]) for m in MODS} for n in DEC}
    G = {(a, b): {m: torch.zeros(n_of[a][m], n_of[b][m],
                                 dtype=torch.float64, device="cuda")
                  for m in MODS} for a, b in NEW_PAIRS}
    T = 0
    t0 = time.time()
    for b in range(NB):
        series = {}
        for n in DEC:
            x = torch.from_numpy(
                np.load(f"{DUMP}/{n}_b{b:03d}.npy")).cuda().float()
            series[n] = {m: x[:, off[n][i]:off[n][i + 1]]
                         for i, m in enumerate(MODS)}
        T += next(iter(series["old"].values())).shape[0]
        for n in DEC:
            for m in MODS:
                c = series[n][m]
                S1[n][m] += c.sum(0).double()
                S2[n][m] += (c * c).sum(0).double()
        for a, bname in NEW_PAIRS:
            for m in MODS:
                G[(a, bname)][m] += (series[a][m].T @ series[bname][m]).double()
        if b % 25 == 0:
            print(f"batch {b + 1}/{NB}, {time.time() - t0:.0f}s", flush=True)

    data = {"T": np.array(T)}
    for n in DEC:
        for m in MODS:
            data[f"{n}|{m}|S1"] = S1[n][m].cpu().numpy()
            data[f"{n}|{m}|S2"] = S2[n][m].cpu().numpy()
    for a, bname in NEW_PAIRS:
        for m in MODS:
            mu_a = S1[a][m] / T
            mu_b = S1[bname][m] / T
            sd_a = torch.sqrt(torch.clamp(S2[a][m] / T - mu_a**2, min=0))
            sd_b = torch.sqrt(torch.clamp(S2[bname][m] / T - mu_b**2, min=0))
            r = (G[(a, bname)][m] / T - torch.outer(mu_a, mu_b)) \
                / torch.outer(sd_a, sd_b)
            r[~torch.isfinite(r)] = torch.nan
            data[f"{a}|{bname}|{m}|r"] = r.cpu().numpy().astype(np.float16)
    np.savez(f"/data/cross_r_new12.npz", **data)
    vol.commit()
    print(f"saved /data/cross_r_new12.npz ({time.time() - t0:.0f}s total)")


@app.function(image=gram_image, gpu="A10G", volumes={"/data": vol},
              timeout=7200, memory=65536)
def grams_f() -> None:
    """Stream all SEVEN dumps batch-aligned once; accumulate S1/S2 + the 6
    F-pair Grams in float64 on GPU; write finished Pearson r (f16)."""
    import time

    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    dec7 = DEC + ["F"]
    alive = {n: _alive(n) for n in dec7}
    n_of = {n: {m: len(alive[n][m]) for m in MODS} for n in dec7}
    off = {n: np.concatenate([[0], np.cumsum([n_of[n][m] for m in MODS])])
           for n in dec7}

    S1 = {n: {m: torch.zeros(n_of[n][m], dtype=torch.float64, device="cuda")
              for m in MODS} for n in dec7}
    S2 = {n: {m: torch.zeros_like(S1[n][m]) for m in MODS} for n in dec7}
    G = {(a, b): {m: torch.zeros(n_of[a][m], n_of[b][m],
                                 dtype=torch.float64, device="cuda")
                  for m in MODS} for a, b in F_PAIRS}
    T = 0
    t0 = time.time()
    for b in range(NB):
        series = {}
        for n in dec7:
            x = torch.from_numpy(
                np.load(f"{DUMP}/{n}_b{b:03d}.npy")).cuda().float()
            series[n] = {m: x[:, off[n][i]:off[n][i + 1]]
                         for i, m in enumerate(MODS)}
        T += next(iter(series["F"].values())).shape[0]
        for n in dec7:
            for m in MODS:
                c = series[n][m]
                S1[n][m] += c.sum(0).double()
                S2[n][m] += (c * c).sum(0).double()
        for a, bname in F_PAIRS:
            for m in MODS:
                G[(a, bname)][m] += (series[a][m].T @ series[bname][m]).double()
        if b % 25 == 0:
            print(f"batch {b + 1}/{NB}, {time.time() - t0:.0f}s", flush=True)

    data = {"T": np.array(T)}
    for n in dec7:
        for m in MODS:
            data[f"{n}|{m}|S1"] = S1[n][m].cpu().numpy()
            data[f"{n}|{m}|S2"] = S2[n][m].cpu().numpy()
    for a, bname in F_PAIRS:
        for m in MODS:
            mu_a = S1[a][m] / T
            mu_b = S1[bname][m] / T
            sd_a = torch.sqrt(torch.clamp(S2[a][m] / T - mu_a**2, min=0))
            sd_b = torch.sqrt(torch.clamp(S2[bname][m] / T - mu_b**2, min=0))
            r = (G[(a, bname)][m] / T - torch.outer(mu_a, mu_b)) \
                / torch.outer(sd_a, sd_b)
            r[~torch.isfinite(r)] = torch.nan
            data[f"{a}|{bname}|{m}|r"] = r.cpu().numpy().astype(np.float16)
    np.savez(f"/data/cross_r_F.npz", **data)
    vol.commit()
    print(f"saved /data/cross_r_F.npz ({time.time() - t0:.0f}s total)")


@app.function(image=gram_image, volumes={"/data": vol}, timeout=3600)
def cleanup() -> None:
    import shutil
    shutil.rmtree(DUMP, ignore_errors=True)
    vol.commit()
    print("removed", DUMP)


@app.local_entrypoint()
def main(stage: str) -> None:
    import concurrent.futures as cf
    if stage == "dumps-trio":
        with cf.ThreadPoolExecutor(3) as ex:
            futs = [ex.submit(dump_old.remote)] + \
                   [ex.submit(dump_new.remote, n) for n in ("newA", "newB")]
            for f in futs:
                f.result()
    elif stage == "dumps-sink":
        list(dump_sink.map(["C", "D", "E"]))
    elif stage == "dump-f":
        dump_sink.remote("F")
    elif stage == "grams":
        grams.remote()
    elif stage == "grams-f":
        grams_f.remote()
    elif stage == "cleanup":
        cleanup.remote()
    else:
        raise SystemExit(f"unknown stage {stage}")
