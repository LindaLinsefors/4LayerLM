"""Modal launcher: pretrain a NEW 4-layer Pile attention-sink LM ("t-freqtest47"),
replicating the co-authors' sink seed-45 recipe (WandB t-87f91319) EXCEPT
  (a) seed 47 instead of 45, and
  (b) the RoPE inverse frequencies are recorded as a TRAJECTORY.

Why: the seed-45/46 sink models were trained with a RoPE spectrum that does not
match their recorded rotary_base 10000 (sink-models/rope_report.md). Leading
hypothesis: the internal pretrain code had inv_freq as a *trainable* parameter.
KEY FINDING while building this launcher: the PUBLIC pretrainer already trains
inv_freq — in `param_decomp.pretrain.models.LlamaSimpleMLP` `inv_freq` is a plain
(non-static) array leaf, `train.py` differentiates and updates every array leaf
(`eqx.filter(..., eqx.is_array)`), and the weight-decay mask is `ndim >= 2` so a
1-D inv_freq rides in the UNdecayed AdamW group. No stop_gradient anywhere.
So NO model/optimizer change is needed: running the public trainer as-is IS the
hypothesized internal behavior. This script only adds
  * inv_freq trajectory recording (every train-loss log, i.e. every `log_every`
    steps) to `<run_dir>/inv_freq_trajectory.npz` + wandb scalars,
  * a fixed pretrain-cache dir `pretrain_cache/spd-t-freqtest47/` (the "spd-"
    prefix keeps it loadable exactly like spd-t-87f91319),
  * Modal plumbing: periodic volume commits, retries + orbax resume.

Config = param_decomp/pretrain/configs/pile_llama_simple_mlp-4L-768-untied-sinks.yaml
(which matches the t-87f91319 WandB dump field-for-field), with seed 47,
data_root /data/sink-models (volume vpd-4layer: datasets/pile_neox_tok_512{,_val}
prestaged by redo-decomps/hide/prestage_data_modal.py), dp: null (one process,
8 local GPUs — initialize_topology(8, 8) would set up the identical (1,8,1) mesh
without jax.distributed anyway), val_data: null exactly as the internal run
(trainer semantics: val_loss from a reseeded schedule over the TRAIN shards).

Run (PowerShell; always PYTHONUTF8=1 PYTHONIOENCODING=utf-8 for modal on Windows):
  modal run own-pretrain/hide/pretrain_modal.py --mode smoke      (~500 steps, 8xH100)
  modal run --detach own-pretrain/hide/pretrain_modal.py --mode full   (100k steps, ~3-6 h)
Relaunching `--mode full` resumes from the latest orbax checkpoint.

Monitor:
  wandb: Linda's default entity, project "param-decomp", group "own-pretrain",
         run id t-freqtest47 (smoke: t-freqtest47-smoke)
  modal volume get vpd-4layer /sink-models/runs/t-freqtest47/inv_freq_trajectory.npz .
  modal volume get vpd-4layer /sink-models/runs/t-freqtest47/metrics.jsonl .
Final model: /data/sink-models/pretrain_cache/spd-t-freqtest47/
  (model_step_100000.safetensors + model_config.yaml, decomposable as usual).
"""

import modal

app = modal.App("own-pretrain-freqtest47")
vol = modal.Volume.from_name("vpd-4layer")

SINK_COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"  # PR #1002 loader/pretrainer
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{SINK_COMMIT}"
    )
)

DATA_ROOT = "/data/sink-models"
FULL_RUN_ID = "t-freqtest47"
SMOKE_RUN_ID = "t-freqtest47-smoke"

# The seed-45 recipe (== configs/pile_llama_simple_mlp-4L-768-untied-sinks.yaml
# == the t-87f91319 WandB config dump), seed 45 -> 47.
BASE_CFG = {
    "run_name": "pile_llama_simple_mlp-4L-768-untied-sinks-freqtest47",
    "seed": 47,
    "dp": None,  # single process, 8 local GPUs (same (1,8,1) mesh as dp: 8)
    "gpus_per_node": 8,
    "dtype": "bfloat16",
    "global_batch": 1024,
    "num_iterations": 100000,
    "warmup_iters": 600,
    "learning_rate": 3e-4,
    "learning_rate_decay_frac": 0.1,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "log_every": 100,
    "val_every": 1000,
    "val_steps": 20,
    "save_every": 1000,
    "keep_last": 2,
    "model": {
        "model_type": "LlamaSimpleMLP",
        "tie_word_embeddings": False,
        "attention_sinks": True,
        "block_size": 512,
        "vocab_size": 50277,
        "n_layer": 4,
        "n_head": 6,
        "n_embd": 768,
        "n_intermediate": 3072,
        "rotary_base": 10000,
        "n_ctx": 512,
        "n_key_value_heads": 6,
        "rms_norm_eps": 1e-6,
    },
    "data": {"kind": "name", "name": "pile_neox_tok_512"},
    "val_data": None,
    "data_root": DATA_ROOT,
    "wandb": {"project": "param-decomp", "group": "own-pretrain"},
}


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=131072,
    volumes={"/data": vol},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb")],
    retries=modal.Retries(max_retries=5, initial_delay=10.0, backoff_coefficient=1.0),
)
def train(cfg_dict: dict, run_id: str) -> str:
    import json
    import threading
    import time
    from pathlib import Path

    import numpy as np

    # ---- durability: commit the volume every 5 min so orbax ckpts + the
    # trajectory survive an unclean container death (resume path relies on it)
    def committer() -> None:
        while True:
            time.sleep(300)
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"volume commit failed (retrying later): {e}", flush=True)

    threading.Thread(target=committer, daemon=True).start()

    import param_decomp.pretrain.train as T
    from param_decomp.pretrain.config import PretrainConfig
    from param_decomp.pretrain.models import _plain_rope_inv_freq

    cfg = PretrainConfig.model_validate({**cfg_dict, "run_id": run_id})
    run_dir = Path(DATA_ROOT) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "inv_freq_trajectory.npz"
    init_inv = np.asarray(_plain_rope_inv_freq(cfg.model), np.float64)  # 10000^(-p/64)

    # ---- trajectory state (reloaded on retry/resume; overlapping steps dropped)
    steps: list[int] = []
    freqs: list[np.ndarray] = []
    losses: list[tuple[int, float]] = []
    if traj_path.exists():
        old = np.load(traj_path)
        steps = [int(s) for s in old["steps"]]
        freqs = [f for f in old["inv_freq"]]
        print(f"loaded existing trajectory: {len(steps)} snapshots", flush=True)

    def save_traj() -> None:
        tmp = traj_path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            steps=np.asarray(steps, np.int64),
            inv_freq=np.asarray(freqs, np.float32),
            init=init_inv.astype(np.float32),
        )
        tmp.replace(traj_path)

    # ---- patch 1: capture the latest TrainState from the jitted step wrapper
    latest: dict = {}
    orig_make_train_step = T.make_train_step

    def make_train_step(cfg_, optimizer):  # noqa: ANN001, ANN202
        inner = orig_make_train_step(cfg_, optimizer)

        def step_fn(state, tokens):  # noqa: ANN001, ANN202
            state, loss = inner(state, tokens)
            latest["state"] = state
            return state, loss

        return step_fn

    T.make_train_step = make_train_step

    # ---- patch 2: snapshot inv_freq at every train-loss log (every log_every steps)
    orig_log = T.MetricsSink.log

    def log(self, step: int, record: dict) -> None:  # noqa: ANN001
        if "train_loss" in record and "state" in latest:
            inv = np.asarray(latest["state"].model.inv_freq, np.float64)
            drift = inv - init_inv
            record = {
                **record,
                "inv_freq_0": float(inv[0]),
                "inv_freq_drift_l2": float(np.linalg.norm(drift)),
                "inv_freq_drift_linf": float(np.abs(drift).max()),
                "inv_freq_logdrift_l2": float(np.linalg.norm(np.log(np.abs(inv) / init_inv))),
            }
            while steps and steps[-1] >= step:  # resume overlap
                steps.pop(), freqs.pop()
            steps.append(step)
            freqs.append(inv.astype(np.float32))
            save_traj()
            losses.append((step, float(record["train_loss"])))
        orig_log(self, step, record)

    T.MetricsSink.log = log

    # ---- patch 3: fixed spd- cache prefix regardless of the wandb project name
    T._cache_dir = lambda cfg_, paths: Path(DATA_ROOT) / "pretrain_cache" / f"spd-{run_id}"

    T._enable_compilation_cache(cfg.paths)
    t0 = time.time()
    T.train(cfg)
    vol.commit()

    # ---- post-run summary + the smoke-critical assertion
    final = np.asarray(freqs[-1], np.float64)
    drift_l2 = float(np.linalg.norm(final - init_inv))
    assert drift_l2 > 1e-5, f"inv_freq did NOT move (drift {drift_l2:.2e}) — trainable patch broken?"
    cache = Path(DATA_ROOT) / "pretrain_cache" / f"spd-{run_id}"
    cache_files = sorted(p.name for p in cache.glob("*"))
    head = losses[:3]
    tail = losses[-3:]
    return json.dumps(
        {
            "run_id": run_id,
            "seconds": round(time.time() - t0),
            "loss_head": head,
            "loss_tail": tail,
            "inv_freq_drift_l2": drift_l2,
            "inv_freq_first4_init": list(init_inv[:4]),
            "inv_freq_first4_final": list(final[:4]),
            "cache_files": cache_files,
            "traj_snapshots": len(steps),
        },
        indent=2,
    )


@app.local_entrypoint()
def main(mode: str = "smoke") -> None:
    if mode == "smoke":
        cfg = {
            **BASE_CFG,
            "run_name": BASE_CFG["run_name"] + "-smoke",
            "num_iterations": 500,
            "log_every": 50,
            "val_every": 250,
            "val_steps": 5,
            "save_every": 250,
        }
        print(train.remote(cfg, SMOKE_RUN_ID))
        return
    assert mode == "full", mode
    print(train.remote(BASE_CFG, FULL_RUN_ID))
