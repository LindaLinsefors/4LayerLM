"""Memory-toy VPD composition root (mirrors param_decomp.experiments.resid_mlp.run).

Loads a pretrained memory-toy target from safetensors, verifies it reproduces its
recorded fact accuracy, and decomposes it through the generic core engine with
neuron-aligned component init. The dataset is the FIXED fact table (train = eval by
design); batches sample pairs uniformly with replacement.

Called with a config yaml (schema below) + data_root + run_id, e.g. from the Modal
launcher (vpd_modal.py). Checkpoints land in <data_root>/runs/<run_id>/ckpts/.
"""

import sys
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import numpy as np
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from pydantic import PositiveInt

import param_decomp.core.hardware_utilization as _hu

# Register the A10G (not in the repo's closed device enumeration; HFU is diagnostic only).
# A10G datasheet: 125 TFLOPS bf16 tensor with sparsity -> dense = 62.5e12.
_A10_KINDS = {"NVIDIA A10", "NVIDIA A10G"}  # Modal A10G containers report either string
if not _A10_KINDS <= _hu.DEVICE_KINDS:
    _hu.DEVICE_KINDS = frozenset(_hu.DEVICE_KINDS | _A10_KINDS)
    _orig_peak = _hu.peak_bf16_dense_flops_per_second
    _hu.peak_bf16_dense_flops_per_second = (
        lambda kind: 62.5e12 if kind in _A10_KINDS else _orig_peak(kind)
    )

from param_decomp.core import placement
from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.components import SiteC
from param_decomp.core.log import setup_logger
from param_decomp.core.model import PlacedModel, Positionless
from param_decomp.core.objective import build_objective
from param_decomp.core.run import (
    Evaluation,
    MetricsSink,
    install_sigterm_flag,
    no_batch_contexts,
    run_decomposition_training,
)
from param_decomp.core.sharding import single_device_mesh
from param_decomp.experiments.config import pin_launch_config, run_instance
from param_decomp.experiments.eval_config import EvalConfig
from typing import Annotated
from pydantic import Field
from param_decomp.core.configs import ExplicitCSpec
from param_decomp.experiments.toy_config import (
    GlobalMlpCiConfig,
    LayerwiseMlpCiConfig,
    ToyExperimentConfig,
    build_toy_ci_arch,
)
from param_decomp.experiments.toy_eval import ToyRun, make_toy_evaluation_operations
from param_decomp.infra.run_files import generate_run_id

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lookup_ci
import memory_toy

lookup_ci.install()  # route init_placed.build_ci_fn through the lookup constructor


class LookupCiConfig(BaseConfig):
    """Trained lookup-table CI: one preactivation per (fact, component) — see
    lookup_ci.py. hidden_dims sizes the vestigial parent MLP (unused output)."""

    type: Literal["lookup"] = "lookup"
    hidden_dims: list[int] = Field(default_factory=lambda: [8])


class MemoryToyDecompositionConfig(BaseConfig):
    sites: ExplicitCSpec
    ci: Annotated[
        LayerwiseMlpCiConfig | GlobalMlpCiConfig | LookupCiConfig,
        Field(discriminator="type"),
    ]


class MemoryToyTargetSchema(BaseConfig):
    arch: Literal["attn", "twoemb", "mix"]
    weights_path: str
    expected_acc: float
    """The target's recorded fact accuracy (from its training summary); the loaded
    forward must reproduce it to 0.01 or the port is broken."""


class MemoryToyDataSchema(BaseConfig):
    facts_path: str
    n_facts: PositiveInt


class MemoryToyExperimentConfig(ToyExperimentConfig):
    target: MemoryToyTargetSchema
    decomposition: MemoryToyDecompositionConfig
    data: MemoryToyDataSchema


MemoryToyRun = ToyRun[memory_toy.MemoryToyRunTarget]


def build_memory_toy_built_run(
    cfg: MemoryToyExperimentConfig, run_id: str, data_root: Path
) -> MemoryToyRun:
    site_cs = memory_toy.canonical_site_cs(
        cfg.target.arch, tuple(SiteC(s.name, s.C) for s in cfg.decomposition.sites.sites)
    )
    build_objective(cfg.pd.loss_metrics, tuple(sc.name for sc in site_cs))
    target = memory_toy.MemoryToyRunTarget(
        arch=cfg.target.arch,
        weights_path=cfg.target.weights_path,
        expected_acc=cfg.target.expected_acc,
        facts_path=cfg.data.facts_path,
        n_facts=cfg.data.n_facts,
        global_batch=cfg.pd.batch_size,
        sites=site_cs,
    )
    return BuiltRun(
        pd=cfg.pd,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id, data_root, None),
        target=target,
        data=None,
        ci_fn=(
            lookup_ci.lookup_ci_arch(cfg.data.n_facts, tuple(cfg.decomposition.ci.hidden_dims))
            if isinstance(cfg.decomposition.ci, LookupCiConfig)
            else build_toy_ci_arch(
                cfg.decomposition.ci,
                tuple(sc.name for sc in site_cs),
                memory_toy.site_specs(cfg.target.arch, site_cs),
            )
        ),
    )


def _verify_target(
    target: memory_toy.MemoryToyTarget, facts: np.ndarray, expected_acc: float
) -> float:
    """The loaded forward must reproduce the trainer's fact accuracy (port check)."""

    @eqx.filter_jit
    def chunk_correct(tgt: memory_toy.MemoryToyTarget, chunk: jax.Array) -> jax.Array:
        logits = memory_toy.clean_logits(tgt, chunk[:, :2])
        return (logits.argmax(-1) == chunk[:, 2]).sum()

    correct = 0
    for start in range(0, len(facts), 16384):
        chunk = jax.numpy.asarray(facts[start : start + 16384])
        correct += int(chunk_correct(target, chunk))
    acc = correct / len(facts)
    assert abs(acc - expected_acc) < 0.01, (
        f"loaded target acc {acc:.4f} != recorded {expected_acc:.4f} — forward port broken?"
    )
    return acc


def run_memory_toy_decomposition(
    built: MemoryToyRun, eval_config: EvalConfig | None, mesh: Mesh
) -> None:
    target_cfg = built.target
    is_main = jax.process_index() == 0

    facts = np.load(target_cfg.facts_path)["facts"][: target_cfg.n_facts].astype(np.int32)
    assert len(facts) == target_cfg.n_facts, (len(facts), target_cfg.n_facts)
    target = memory_toy.load_memory_toy_target(target_cfg.weights_path, target_cfg.arch)
    acc = _verify_target(target, facts, target_cfg.expected_acc)
    if is_main:
        print(f"target {target_cfg.arch} verified: fact acc {acc:.4f} "
              f"(recorded {target_cfg.expected_acc:.4f})", flush=True)

    model = memory_toy.replicate_target(
        memory_toy.memory_toy_decomposed_model(target_cfg.arch, target, target_cfg.sites), mesh
    )
    placed_model = PlacedModel(
        model=model, placement=placement.from_config("ddp", mesh, model.sites)
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)
    table = jax.device_put(
        jax.numpy.asarray(facts[:, :2]), NamedSharding(mesh, P())
    )

    # `table` is a filter_jit ARG (traced, not baked into the HLO).
    def make_sampler(batch_size: int):
        @eqx.filter_jit
        def sample(table: jax.Array, step_key: jax.Array) -> jax.Array:
            idx = random.randint(step_key, (batch_size,), 0, table.shape[0])
            tokens = jax.numpy.concatenate([table[idx], idx[:, None].astype(table.dtype)], 1)
            return jax.sharding.reshard(
                tokens, NamedSharding(mesh, P(placement.batch_axes(mesh)))
            )

        return sample

    sample_train = make_sampler(target_cfg.global_batch)

    def sample_batch(step: int) -> jax.Array:
        return sample_train(table, random.fold_in(data_key, step))

    operations = []
    if eval_config is not None:
        eval_sampler = make_sampler(eval_config.batch_size)
        operations.extend(
            make_toy_evaluation_operations(
                eval_config,
                built.pd.seed,
                compiler_options={},
                model=placed_model,
                ci_capture_keys=built.ci_fn.capture_keys,
                mesh=mesh,
                sample_eval_batch=lambda index: eval_sampler(
                    table, random.fold_in(data_key, built.pd.steps + index)
                ),
                probe_ci=lambda state: {},  # only read by UVPlots, which we don't configure
                wandb_configured=built.run.wandb is not None,
            )
        )
    evaluation = Evaluation(tuple(operations), lambda invocation: invocation, no_batch_contexts)

    sink = MetricsSink.for_run(built.run, is_main)
    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        model=placed_model,
        ci_fn=built.ci_fn,
        positions=Positionless(),
        remat_recon_forwards=False,
        remat_ci_fn=False,
        compiler_options={},
        sample_batch=sample_batch,
        evaluation=evaluation,
        sink=sink,
        profiling=None,
        component_initializer=memory_toy.neuron_aligned_initializer,
    )


def main(
    config: str,
    data_root: Path,
    run_id: str | None = None,
) -> None:
    schema_raw = yaml.safe_load(Path(config).read_text())
    data_root = Path(data_root)
    if run_id is None:
        run_id = generate_run_id("param_decomp")
    cfg = MemoryToyExperimentConfig(**schema_raw)
    built = build_memory_toy_built_run(cfg, run_id, data_root)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    pin_launch_config(built.run.run_dir, yaml.safe_dump(schema_raw, sort_keys=False))
    install_sigterm_flag()
    mesh = single_device_mesh()
    run_memory_toy_decomposition(built, cfg.eval, mesh)
