"""Modal launcher for the corrected-RoPE C-analogue decomposition (redo-decomps).

Trains config_ropefix_C.yaml (byte-faithful C recipe, see the yaml header) with the
public param-decomp trainer at the sink-loader commit, with
`param_decomp.targets.llama_simple_mlp.plain_rope_inv_freq` monkeypatched to the
FITTED spectrum (sink-models/hide/cache/fitted_freqs_avg.npz) before the target is
built -- the same proven patch path as components/hide/typical_act_modal_C.py
(open_jax_run and the trainer share the target builder).

Modes:
  modal run redo-decomps/hide/train_modal.py --mode check
      config-parse + build-validation only (CPU container, no GPU).
  modal run redo-decomps/hide/train_modal.py --mode smoke
      600-step run on 8xH100 (run_id p-ropefix-smoke): first attempt arms a
      watchdog that HARD-KILLS the container ~30 s after the first checkpoint
      lands, Modal retries, the trainer must resume from the checkpoint and
      finish -- the exact preemption path the full run relies on.
  modal run --detach redo-decomps/hide/train_modal.py --mode full
      the real run (run_id p-ropefix-c45, 100k steps, ~8 h). Relaunching with
      the same command resumes from the latest checkpoint (stable run_id +
      byte-identical config).

Data root /data/sink-models on volume vpd-4layer: pretrain_cache/spd-t-87f91319
(the frozen target) + datasets/pile_neox_tok_512{,_val} (prestage_data_modal.py).
Checkpoints land in /data/sink-models/runs/<run_id>/ckpts. WandB: secret `wandb`,
Linda's default entity, project param-decomp.
"""

import modal

app = modal.App("ropefix-train")
vol = modal.Volume.from_name("vpd-4layer")

SINK_COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{SINK_COMMIT}"
    )
)

DATA_ROOT = "/data/sink-models"
# run ids must match the repo's p-<8hex> pattern; mnemonic: c45e = "C-analogue,
# seed-45 target", 0002 = smoke / 0001 = the real run. (p-c45e0000 was a dead
# smoke attempt with the broken optax-muon default; its run dir can be deleted.)
SMOKE_RUN_ID = "p-c45e0007"
FULL_RUN_ID = "p-c45e0001"

# The winning substrate (smoke p-c45e000a, 2026-09-18): TUNED_V1_COMPILER_OPTIONS
# + xla_gpu_use_memcpy_local_p2p -- lowers single-process intra-node all-to-alls
# as cudaMemcpy instead of NCCL grouped send/recv, whose P2P transport Modal's
# containers block (ncclAlltoAll "unhandled system error" otherwise).
MEMCPY_COMPILER_OPTIONS = {
    "xla_gpu_enable_latency_hiding_scheduler": True,
    "xla_gpu_enable_triton_gemm": False,
    "xla_gpu_enable_command_buffer": "",
    "xla_gpu_enable_highest_priority_async_stream": True,
    "xla_gpu_all_reduce_combine_threshold_bytes": 1073741824,
    "xla_gpu_all_gather_combine_threshold_bytes": 1073741824,
    "xla_gpu_reduce_scatter_combine_threshold_bytes": 134217728,
    "xla_gpu_enable_pipelined_all_gather": True,
    "xla_gpu_enable_pipelined_reduce_scatter": True,
    "xla_gpu_enable_pipelined_all_reduce": True,
    "xla_gpu_enable_while_loop_double_buffering": True,
    "xla_gpu_enable_all_gather_combine_by_dim": False,
    "xla_gpu_enable_reduce_scatter_combine_by_dim": False,
    "xla_gpu_use_memcpy_local_p2p": True,
}


def _bootstrap_env(cfg_text: str) -> None:
    """Mimic experiments/lm/run.py: launch_env into os.environ BEFORE importing jax."""
    import os

    import yaml

    from param_decomp.experiments.lm.runtime import RuntimeConfig

    runtime = RuntimeConfig.model_validate(yaml.safe_load(cfg_text)["runtime"])
    os.environ.update(runtime.launch_env.as_env())


def _install_rope_patch(log_inv_freq: list) -> None:
    import jax.numpy as jnp
    import numpy as np

    from param_decomp.targets import llama_simple_mlp as lsm

    fitted = jnp.asarray(np.exp(np.asarray(log_inv_freq, np.float64)), jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: fitted
    assert lsm.plain_rope_inv_freq(None) is fitted


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def check(cfg_text: str, log_inv_freq: list) -> str:
    """Parse + resolve the config end-to-end on CPU (no training)."""
    from pathlib import Path

    _bootstrap_env(cfg_text)
    _install_rope_patch(log_inv_freq)
    Path("/tmp/config.yaml").write_text(cfg_text)
    from param_decomp.experiments.lm.config import load_config

    built, authored = load_config(Path("/tmp/config.yaml"), "p-00c0ffee", Path(DATA_ROOT))
    sites = built.target.sites
    total_c = sum(s.C for s in sites)
    from param_decomp.infra.dataset_store import read_dataset_meta

    meta = read_dataset_meta(built.data.dir)
    eval_meta = read_dataset_meta(built.data.eval_dir)
    return (f"config OK: {len(sites)} sites, total C {total_c}, steps {built.pd.steps}, "
            f"batch {built.pd.batch_size}, seed {built.pd.seed}; "
            f"train meta {meta}, eval meta {eval_meta}; "
            f"run_dir {built.run.run_dir}")


image_nccl221 = image.pip_install("nvidia-nccl-cu12==2.21.5")

_train_kw = dict(gpu="H100:8", cpu=32, memory=131072, volumes={"/data": vol},
                 timeout=86400, secrets=[modal.Secret.from_name("wandb")],
                 retries=modal.Retries(max_retries=5, initial_delay=10.0,
                                       backoff_coefficient=1.0))


@app.function(image=image_nccl221, **_train_kw)
def train_nccl221(cfg_text: str, log_inv_freq: list, run_id: str,
                  kill_after_first_ckpt: bool) -> str:
    """Same trainer on an image with NCCL pinned to 2.21.5 (pre-cuMem alltoall)."""
    return _train_body(cfg_text, log_inv_freq, run_id, kill_after_first_ckpt)


@app.function(image=image, **_train_kw)
def train(cfg_text: str, log_inv_freq: list, run_id: str, kill_after_first_ckpt: bool) -> str:
    return _train_body(cfg_text, log_inv_freq, run_id, kill_after_first_ckpt)


def _train_body(cfg_text: str, log_inv_freq: list, run_id: str,
                kill_after_first_ckpt: bool) -> str:
    import os
    import threading
    import time
    from pathlib import Path

    Path("/tmp/config.yaml").write_text(cfg_text)
    run_dir = Path(DATA_ROOT) / "runs" / run_id
    ckpt_dir = run_dir / "ckpts"
    marker = run_dir / "smoke_kill_done.marker"

    # periodic volume commit so checkpoints survive an unclean container death
    def committer():
        while True:
            time.sleep(300)
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"volume commit failed (retrying later): {e}", flush=True)

    threading.Thread(target=committer, daemon=True).start()

    if kill_after_first_ckpt and not marker.exists():
        def watchdog():
            while True:
                time.sleep(20)
                steps = [p for p in ckpt_dir.glob("*")
                         if p.name.isdigit() and int(p.name) >= 200] \
                    if ckpt_dir.exists() else []
                if steps:
                    print(f"WATCHDOG: checkpoint {sorted(steps)[-1].name} seen; "
                          "killing container in 30 s to test resume", flush=True)
                    time.sleep(30)
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("killed once\n")
                    vol.commit()
                    os._exit(43)

        threading.Thread(target=watchdog, daemon=True).start()
        print("smoke watchdog armed (first attempt)", flush=True)

    _bootstrap_env(cfg_text)
    _install_rope_patch(log_inv_freq)
    from param_decomp.experiments.lm.training import main as train_main

    t0 = time.time()
    train_main(Path("/tmp/config.yaml"), Path(DATA_ROOT), 8, run_id)
    vol.commit()
    steps = sorted(int(p.name) for p in ckpt_dir.glob("*") if p.name.isdigit())
    return f"done in {time.time() - t0:.0f}s; checkpoints at steps {steps}"


@app.local_entrypoint()
def main(mode: str = "check") -> None:
    from pathlib import Path

    import numpy as np
    import yaml

    here = Path(__file__).resolve().parent
    cfg_text = (here / "config_ropefix_C.yaml").read_text()
    lif = np.load(here.parents[1] / "sink-models" / "hide" / "cache"
                  / "fitted_freqs_avg.npz")["log_inv_freq"].tolist()

    if mode == "check":
        print(check.remote(cfg_text, lif))
        return

    if mode.startswith("smoke"):
        cfg = yaml.safe_load(cfg_text)
        cfg["run_name"] += "-smoke"
        cfg["pd"]["steps"] = 600
        cfg["cadence"]["train_log_every"] = 50
        cfg["cadence"]["checkpointing"]["save_every"] = 200
        cfg["eval"]["every"] = 200
        run_id, fn = SMOKE_RUN_ID, train
        if mode == "smoke-tp2":
            # parallel memory-shape variant: shard the C axis 2-way instead of
            # batch/d 8-way (SPEC D4: same trajectory up to float reassociation)
            cfg["runtime"]["fsdp"] = 4
            cfg["runtime"]["tp"] = 2
            run_id = "p-c45e0005"
        elif mode == "smoke-zero1":
            # dodge the owner-layout NS-staging all-to-all: zero1 stages via
            # all-gathers ("~equivalent comms under elementwise optimizers")
            cfg["runtime"]["sharding"] = "zero1"
            run_id = "p-c45e0008"
        elif mode == "smoke-nccl221":
            run_id, fn = "p-c45e0009", train_nccl221
        elif mode == "smoke-memcpy":
            cfg["runtime"]["compiler_options"] = dict(MEMCPY_COMPILER_OPTIONS)
            run_id = "p-c45e000a"
        elif mode == "smoke-nop2p":
            # force NCCL send/recv off the blocked P2P transport (staged via SHM)
            cfg["runtime"]["launch_env"]["env"]["NCCL_P2P_DISABLE"] = "1"
            run_id = "p-c45e000b"
        smoke_text = yaml.safe_dump(cfg, sort_keys=False)
        print(fn.remote(smoke_text, lif, run_id, True))
        return

    assert mode == "full", mode
    cfg = yaml.safe_load(cfg_text)
    cfg["runtime"]["compiler_options"] = MEMCPY_COMPILER_OPTIONS
    print(train.remote(yaml.safe_dump(cfg, sort_keys=False), lif, FULL_RUN_ID, False))
