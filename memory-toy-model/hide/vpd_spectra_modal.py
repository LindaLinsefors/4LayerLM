"""Extract per-component mean CI (lower_leaky) from the memory-toy VPD checkpoints.

For each chosen decomposition (vpd_report.md table): restore V/U + CI fn from the run's
latest orbax checkpoint, run the CI function over the FULL fact table, and return the
per-site per-component mean CI. Saved to hide/cache/vpd_mean_ci.npz with keys
"<arch>|<k>|<site>" -> (C,) float64, plus "<arch>|<k>|acc" scalars for reference.

Run (PYTHONUTF8=1):  modal run memory-toy-model/hide/vpd_spectra_modal.py
Then:                python memory-toy-model/hide/plot_vpd_spectra.py
"""

import json
from pathlib import Path

import modal

from vpd_modal import ARCH_SITES, SITE_C, image, stable_run_id, vol

image = image.add_local_python_source("vpd_modal")

app = modal.App("memory-toy-vpd-spectra")

DATA_ROOT = "/data/memory-toy/vpd"

# the report's chosen cell -> coefficient (run key = final-<arch>-k<k>-imp<coeff:g>)
CHOSEN = {
    ("attn", 15): 1e-4, ("attn", 16): 1e-3, ("attn", 17): 1e-5,
    ("twoemb", 15): 1e-3, ("twoemb", 16): 1e-3, ("twoemb", 17): 1e-3,
    ("mix", 15): 1e-4, ("mix", 16): 1e-3, ("mix", 17): 1e-5,
}
# lookup-table-CI diagnostic runs (run key = lookup-<arch>-k<k>-imp<coeff:g>)
LOOKUP = {("attn", 15): 1e-4, ("twoemb", 15): 1e-3, ("mix", 15): 1e-4}


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=1800)
def mean_ci(arch: str, k: int, coeff: float, ci: str = "mlp", c_all: int | None = None) -> dict:
    import sys

    sys.path.insert(0, "/root/vpd")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import random

    import memory_toy
    from param_decomp.core.checkpoint import (
        make_read_only_checkpoint_manager,
        restore_decomposition,
    )
    from param_decomp.core.ci_fn import build_ci_fn
    from param_decomp.core.components import SiteC, init_component_stacks, require_full_emission
    from param_decomp.core.train import Decomposition
    from param_decomp.experiments.toy_config import GlobalMlpCiConfig, build_toy_ci_arch

    if ci == "lookup" and c_all == 8192:  # the C>n_facts series (k12)
        run_key = f"lookup4-{arch}-k{k}-imp{coeff:g}-C{c_all}"
    elif ci == "lookup":
        run_key = f"lookup3-{arch}-k{k}-imp{coeff:g}"
    else:
        run_key = f"final-{arch}-k{k}-imp{coeff:g}"
    run_dir = Path(DATA_ROOT) / "runs" / stable_run_id(run_key)
    assert (run_dir / "ckpts").exists(), run_dir

    site_cs = tuple(SiteC(name, c_all or SITE_C[name]) for name in ARCH_SITES[arch])
    sites = memory_toy.site_specs(arch, site_cs)
    suffix = {"attn": "", "twoemb": "-twoemb", "mix": "-mix"}[arch]
    target = memory_toy.load_memory_toy_target(
        f"/data/memory-toy/runs/f2e{k}{suffix}/model_final.safetensors", arch
    )
    model = memory_toy.memory_toy_decomposed_model(arch, target, site_cs)

    if ci == "lookup":
        import lookup_ci
        reference_ci = lookup_ci.build_lookup_ci_fn(
            lookup_ci.lookup_ci_arch(2**k), sites, random.PRNGKey(1)
        )
    else:
        arch_cfg = GlobalMlpCiConfig(type="global_mlp", hidden_dims=[2048, 2048])
        ci_arch = build_toy_ci_arch(arch_cfg, tuple(s.name for s in sites), sites)
        reference_ci = build_ci_fn(ci_arch, sites, random.PRNGKey(1))
    reference = Decomposition(
        components=init_component_stacks(sites, random.PRNGKey(0)),
        ci_fn=reference_ci,
    )
    mgr = make_read_only_checkpoint_manager(run_dir / "ckpts")
    step = mgr.latest_step()
    decomposition = restore_decomposition(mgr, step, reference)
    ci_fn = decomposition.ci_fn

    facts = np.load("/data/memory-toy/facts_seed0.npz")["facts"][: 2**k].astype(np.int32)

    @jax.jit
    def batch_ci_sums(tokens: jax.Array) -> dict:
        # tokens: (B,3) = (t0, t1, fact_idx); the mlp CI fn's taps ignore column 2
        taps = model.clean_forward(tokens, ci_fn.capture_keys, placement=None).captures
        ci = ci_fn(taps, remat=False, placement=None)
        return {
            site: jnp.sum(require_full_emission(v).astype(jnp.float32), axis=0)
            for site, v in ci.lower.items()
        }

    sums = {s.name: np.zeros(s.C, dtype=np.float64) for s in sites}
    for start in range(0, len(facts), 8192):
        rows = facts[start : start + 8192]
        chunk = jnp.asarray(
            np.column_stack([rows[:, :2], np.arange(start, start + len(rows), dtype=np.int32)])
        )
        for site, v in batch_ci_sums(chunk).items():
            sums[site] += np.asarray(v)

    return {
        "step": int(step),
        **{site: (total / len(facts)).tolist() for site, total in sums.items()},
    }


@app.local_entrypoint()
def main(which: str = "mlp"):
    import numpy as np

    here = Path(__file__).resolve().parent
    if which == "lookup4":
        jobs = [(arch, 12, coeff, "lookup", 8192)
                for arch in ("attn", "twoemb", "mix") for coeff in (1e-4, 1e-5, 1e-6)]
        cache = here / "cache" / "vpd_mean_ci_lookup4.npz"
    elif which == "lookup":
        jobs = [(arch, k, coeff, "lookup", 1600) for (arch, k), coeff in LOOKUP.items()]
        cache = here / "cache" / "vpd_mean_ci_lookup.npz"
    else:
        jobs = [(arch, k, coeff, "mlp", None) for (arch, k), coeff in CHOSEN.items()]
        cache = here / "cache" / "vpd_mean_ci.npz"
    results = list(mean_ci.starmap(jobs))
    out = {}
    for (arch, k, coeff, _ci, _c), res in zip(jobs, results):
        prefix = f"{arch}|{k}|" if which != "lookup4" else f"{arch}|{coeff:g}|"
        for site, values in res.items():
            if site == "step":
                continue
            out[prefix + site] = np.array(values)
        print(arch, k, "step", res["step"],
              {s: int((np.array(v) > 1e-6).sum()) for s, v in res.items() if s != "step"})
    np.savez_compressed(cache, **out)
    print("saved", cache)
