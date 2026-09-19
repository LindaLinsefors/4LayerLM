"""Modal launcher for the corrected-RoPE C-analogue decomposition (redo-decomps).

Trains config_ropefix_C.yaml (byte-faithful C recipe, see the yaml header) with the
public param-decomp trainer at the sink-loader commit, with
`param_decomp.targets.llama_simple_mlp.plain_rope_inv_freq` monkeypatched to the
FITTED spectrum (sink-models/hide/cache/fitted_freqs_avg.npz) before the target is
built -- the same proven patch path as components/hide/typical_act_modal_C.py
(open_jax_run and the trainer share the target builder).

Modes:
  modal run sink-models/redo-decomps/hide/train_modal.py --mode check
      config-parse + build-validation only (CPU container, no GPU).
  modal run sink-models/redo-decomps/hide/train_modal.py --mode smoke
      600-step run on 8xH100 (run_id p-ropefix-smoke): first attempt arms a
      watchdog that HARD-KILLS the container ~30 s after the first checkpoint
      lands, Modal retries, the trainer must resume from the checkpoint and
      finish -- the exact preemption path the full run relies on.
  modal deploy sink-models/redo-decomps/hide/train_modal.py     (once, + after edits here)
  modal run sink-models/redo-decomps/hide/train_modal.py --mode full
      the real run (run_id p-c45e0001, 100k steps, ~14 h at 0.49 s/step).
      Relaunching resumes from the latest checkpoint (stable run_id +
      byte-identical config).

LAUNCH PATTERN CHANGE (2026-09-19, intentional -- do not "fix" back):
  --mode full no longer trains inside this client. It .spawn()s `train` on the
  DEPLOYED app and exits immediately; the run is fully server-side. The old
  `modal run --detach ... --mode full` is DEPRECATED for this script: --detach
  still leaves an attached local client, and its death killed two long runs
  (~8 h JWT expiry `AuthError: Jwt is expired`, and session-cleanup teardown) --
  see CLAUDE.md substrate-recipe item 5. check/smoke stay attached on purpose.
  Notes:
  * Deploy BEFORE the first full launch and after any edit to this file
    (spawn runs the code as of the last `modal deploy`, not your working copy).
  * Redeploying does NOT kill an in-flight run (it keeps its old version).
  * Do NOT launch --mode full while a full run is already in flight: a second
    container would resume from the same run dir's checkpoints and race the
    live one. Check first: `modal app list` / the wandb run still stepping.
  * As of 2026-09-19 the app is ALREADY deployed and the live F run
    (p-c45e0001) is a task on it (`modal app list`: ropefix-train, deployed,
    1 task) -- it was respawned server-side this morning with an ad-hoc spawn
    before this entrypoint existed. Leave it alone; if it dies, relaunch with
    `--mode full` (which now does exactly that spawn).

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
# decomposition "G": C's recipe applied to Linda's own t-freqtest47 model
# (config_G_freqtest47.yaml); 47de = "seed-47 target, decomposition"
G_RUN_ID = "p-47de0001"

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


def _install_rope_patch(inv_freq: list) -> None:
    """inv_freq: RAW per-plane frequencies (not log -- learned spectra can carry
    negative near-zero planes, e.g. 6 of t-freqtest47's unconstrained tail)."""
    import jax.numpy as jnp
    import numpy as np

    from param_decomp.targets import llama_simple_mlp as lsm

    spec = jnp.asarray(np.asarray(inv_freq, np.float64), jnp.float32)
    lsm.plain_rope_inv_freq = lambda cfg: spec
    assert lsm.plain_rope_inv_freq(None) is spec


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def check(cfg_text: str, inv_freq: list) -> str:
    """Parse + resolve the config end-to-end on CPU (no training)."""
    from pathlib import Path

    _bootstrap_env(cfg_text)
    _install_rope_patch(inv_freq)
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
def train_nccl221(cfg_text: str, inv_freq: list, run_id: str,
                  kill_after_first_ckpt: bool) -> str:
    """Same trainer on an image with NCCL pinned to 2.21.5 (pre-cuMem alltoall)."""
    return _train_body(cfg_text, inv_freq, run_id, kill_after_first_ckpt)


@app.function(image=image, **_train_kw)
def train(cfg_text: str, inv_freq: list, run_id: str, kill_after_first_ckpt: bool) -> str:
    return _train_body(cfg_text, inv_freq, run_id, kill_after_first_ckpt)


def _train_body(cfg_text: str, inv_freq: list, run_id: str,
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
    _install_rope_patch(inv_freq)
    from param_decomp.experiments.lm.training import main as train_main

    t0 = time.time()
    train_main(Path("/tmp/config.yaml"), Path(DATA_ROOT), 8, run_id)
    vol.commit()
    steps = sorted(int(p.name) for p in ckpt_dir.glob("*") if p.name.isdigit())
    return f"done in {time.time() - t0:.0f}s; checkpoints at steps {steps}"


waiter_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")


@app.function(image=waiter_image, volumes={"/data": vol}, timeout=86400)
def wait_and_spawn_g(cfg_text: str) -> str:
    """Server-side hand-off: poll the volume until the freqtest47 pretrain has
    exported its final model + full inv_freq trajectory, then spawn the G
    decomposition on this deployed app with the LEARNED spectrum. Runs in a
    cheap CPU container so the hand-off survives laptop/session death."""
    import time
    from pathlib import Path

    import numpy as np

    cache = Path(DATA_ROOT) / "pretrain_cache" / "spd-t-freqtest47"
    traj = Path(DATA_ROOT) / "runs" / "t-freqtest47" / "inv_freq_trajectory.npz"
    while True:
        vol.reload()
        if (cache / "model_step_100000.safetensors").exists()                 and (cache / "model_config.yaml").exists() and traj.exists():
            d = np.load(traj)
            if int(d["steps"].max()) >= 100000:
                break
        print("waiting for freqtest47 completion...", flush=True)
        time.sleep(300)

    d = np.load(traj)
    final = np.asarray(d["inv_freq"][int(np.argmax(d["steps"]))], np.float64)
    init = np.asarray(d["init"], np.float64)
    drift = float(np.linalg.norm(final - init))
    assert drift > 0.1, f"final spectrum barely moved (drift {drift:.3f}) -- wrong file?"
    print(f"freqtest47 complete; final omega0 {final[0]:.4f}, drift L2 {drift:.3f}; "
          "spawning G", flush=True)
    call = modal.Function.from_name("ropefix-train", "train").spawn(
        cfg_text, final.tolist(), G_RUN_ID, False)
    return f"spawned G ({G_RUN_ID}): {call.object_id}"


@app.local_entrypoint()
def main(mode: str = "check") -> None:
    from pathlib import Path

    import numpy as np
    import yaml

    here = Path(__file__).resolve().parent
    cfg_text = (here / "config_ropefix_C.yaml").read_text()
    # here = <root>/sink-models/redo-decomps/hide -> parents[1] = sink-models
    lif = np.exp(np.load(here.parents[1] / "hide" / "cache"
                         / "fitted_freqs_avg.npz")["log_inv_freq"]).tolist()

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

    if mode == "spawn-g-waiter":
        cfg_g = yaml.safe_load((here / "config_G_freqtest47.yaml").read_text())
        g_text = yaml.safe_dump(cfg_g, sort_keys=False)
        call = modal.Function.from_name(app.name, "wait_and_spawn_g").spawn(g_text)
        print(f"waiter spawned server-side: {call.object_id}")
        print("It will launch G ({}) automatically when freqtest47 finishes.".format(G_RUN_ID))
        return

    assert mode == "full", mode
    cfg = yaml.safe_load(cfg_text)
    cfg["runtime"]["compiler_options"] = MEMCPY_COMPILER_OPTIONS
    full_text = yaml.safe_dump(cfg, sort_keys=False)
    # Server-side launch (see docstring): spawn on the DEPLOYED app so no local
    # client stays attached. train.spawn() here would NOT be safe -- under
    # `modal run` the app is ephemeral and dies with this process.
    try:
        call = modal.Function.from_name(app.name, "train").spawn(
            full_text, lif, FULL_RUN_ID, False)
    except modal.exception.NotFoundError:
        raise SystemExit(
            f"app '{app.name}' is not deployed yet -- run:\n"
            f"  modal deploy sink-models/redo-decomps/hide/train_modal.py\nthen retry --mode full."
        )
    print(f"spawned server-side; function call id: {call.object_id}")
    print("This terminal/client may now close or die -- the run is unaffected.")
    print(f"Monitor: wandb (project param-decomp, run *-ropefix), or: modal app logs {app.name}")
    print("Fetch the final summary later with:")
    print(f"  python -c \"import modal; print(modal.FunctionCall.from_id('{call.object_id}').get())\"")
