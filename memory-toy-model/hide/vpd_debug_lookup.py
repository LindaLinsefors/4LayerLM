"""Debug: restore a lookup-CI checkpoint and print table stats + recomputed L0."""

from pathlib import Path

import modal

from vpd_modal import ARCH_SITES, SITE_C, image, stable_run_id, vol

image = image.add_local_python_source("vpd_modal")
app = modal.App("memory-toy-vpd-debug")
DATA_ROOT = "/data/memory-toy/vpd"


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=900)
def debug(run_key: str, arch: str, k: int) -> None:
    import sys

    sys.path.insert(0, "/root/vpd")
    import jax.numpy as jnp
    import numpy as np
    from jax import random

    import lookup_ci
    import memory_toy
    from param_decomp.core.checkpoint import (
        make_read_only_checkpoint_manager,
        restore_decomposition,
    )
    from param_decomp.core.components import SiteC, init_component_stacks
    from param_decomp.core.train import Decomposition

    run_dir = Path(DATA_ROOT) / "runs" / stable_run_id(run_key)
    site_cs = tuple(SiteC(name, SITE_C[name]) for name in ARCH_SITES[arch])
    sites = memory_toy.site_specs(arch, site_cs)
    reference = Decomposition(
        components=init_component_stacks(sites, random.PRNGKey(0)),
        ci_fn=lookup_ci.build_lookup_ci_fn(lookup_ci.lookup_ci_arch(2**k), sites, random.PRNGKey(1)),
    )
    mgr = make_read_only_checkpoint_manager(run_dir / "ckpts")
    step = mgr.latest_step()
    print("latest step:", step, "all steps:", mgr.all_steps())
    dec = restore_decomposition(mgr, step, reference)
    for name, table in zip(dec.ci_fn.output_names, dec.ci_fn.tables, strict=True):
        t = np.asarray(table)
        print(f"{name}: shape {t.shape} min {t.min():.4f} mean {t.mean():.4f} "
              f"max {t.max():.4f} frac>0 {(t > 0).mean():.4f} frac==1 {(t == 1.0).mean():.4f}")
    # recompute L0 on the first 2048 facts through the restored ci_fn
    facts = np.load("/data/memory-toy/facts_seed0.npz")["facts"][: 2**k].astype(np.int32)
    rows = facts[:2048]
    tokens = jnp.asarray(np.column_stack([rows[:, :2], np.arange(2048, dtype=np.int32)]))
    target = memory_toy.load_memory_toy_target(
        f"/data/memory-toy/runs/f2e{k}{'' if arch == 'attn' else '-' + arch}/model_final.safetensors",
        arch,
    )
    model = memory_toy.memory_toy_decomposed_model(arch, target, site_cs)
    taps = model.clean_forward(tokens, dec.ci_fn.capture_keys, placement=None).captures
    ci = dec.ci_fn(taps, remat=False, placement=None)
    l0 = sum(float((np.asarray(v) > 0).sum(-1).mean()) for v in ci.lower.values())
    print("recomputed L0/datapoint (rows 0..2047):", l0)
    # per-row-index L0 profile straight off the tables
    per_row = sum((np.asarray(t) > 0).sum(-1) for t in dec.ci_fn.tables)
    n = per_row.shape[0]
    for lo, hi in [(0, 2048), (n // 2 - 1024, n // 2 + 1024), (n - 2048, n)]:
        print(f"table rows [{lo}:{hi}]: mean per-fact L0 {per_row[lo:hi].mean():.1f}")
    print("global mean per-fact L0 from tables:", per_row.mean())
    print("rows with L0 < 300:", int((per_row < 300).sum()), "of", n)


@app.local_entrypoint()
def main():
    debug.remote("lookup-twoemb-k15-imp0.001", "twoemb", 15)
