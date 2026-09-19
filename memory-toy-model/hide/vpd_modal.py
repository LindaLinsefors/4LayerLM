"""VPD decompositions of the memory-toy models on Modal.

One A10G container per run; param-decomp (public, latest main 5ffc73d7) installed from
GitHub; our memory_toy target/composition-root modules shipped from hide/vpd/. Targets +
fact table live on volume vpd-4layer (/data/memory-toy). VPD run dirs land on the volume
under /data/memory-toy/vpd/runs/<run_id>/ (run ids are stable hashes of the run key, so
relaunching resumes from checkpoints).

Recipe: C-recipe loss stack, toy-sized (see the hyperparameter plan in the chat log /
CLAUDE.md): ImportanceMinimalityLoss coeff swept in the pilot, frequency = coeff/3,
NonlinearityLocality 3e-5 (neuron units), MergedStochasticSubsetPPGD coeff 1.0
adv_fraction 0.5, Faithfulness 1000; AdamW 2e-3 cosine; C = 192 (96x96 sites) / 768
(MLP sites); global_mlp CI fn [2048, 2048]; batch 2048, 30k steps; neuron-aligned init.

Run (PYTHONUTF8=1):
  modal run memory-toy-model/hide/vpd_modal.py --mode smoke
  modal run --detach memory-toy-model/hide/vpd_modal.py --mode pilot
  modal run --detach memory-toy-model/hide/vpd_modal.py --mode final --coeff 1e-4
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("memory-toy-vpd")

COMMIT = "5ffc73d79b2dee374c406396efa832023ae5eee7"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
    # the [cuda12] extra on the direct URL silently failed to install CUDA jaxlib
    # (containers fell back to CPU) — install it explicitly, matching the pyproject pin
    .pip_install("jax[cuda12]==0.11.1")
    .add_local_dir(Path(__file__).resolve().parent / "vpd", remote_path="/root/vpd")
)
vol = modal.Volume.from_name("vpd-4layer")

DATA_ROOT = "/data/memory-toy/vpd"
FACTS_PATH = "/data/memory-toy/facts_seed0.npz"

ATTN_SITES = ["q_proj", "k_proj", "v_proj", "o_proj"]
SITE_C = {"q_proj": 192, "k_proj": 192, "v_proj": 192, "o_proj": 192,
          "mix": 192, "c_fc": 768, "down_proj": 768}
ARCH_SITES = {
    "attn": ATTN_SITES + ["c_fc", "down_proj"],
    "twoemb": ["c_fc", "down_proj"],
    "mix": ["mix", "c_fc", "down_proj"],
}

COSINE = lambda peak: {
    "max_val": peak,
    "points": [{"at": 0.0, "frac": 1.0}, {"at": 1.0, "frac": 0.1, "interp": "cosine"}],
}


def build_config(arch: str, k: int, imp_coeff: float, expected_acc: float,
                 steps: int = 30_000, save_every: int = 5_000, ci: str = "mlp",
                 c_all: int | None = None) -> dict:
    suffix = {"attn": "", "twoemb": "-twoemb", "mix": "-mix"}[arch]
    return {
        "run_name": f"memtoy-{arch}-k{k}-imp{imp_coeff:g}",
        "cadence": {
            "train_log_every": 200,
            "checkpointing": {
                "kind": "periodic", "save_every": save_every,
                "retention": {"kind": "keep_last", "n": 2},
            },
        },
        "decomposition": {
            "sites": {
                "kind": "explicit",
                "sites": [{"name": name, "C": c_all or SITE_C[name]}
                          for name in ARCH_SITES[arch]],
            },
            "ci": ({"type": "lookup"} if ci == "lookup"
                   else {"type": "global_mlp", "hidden_dims": [2048, 2048]}),
        },
        "pd": {
            "seed": 0,
            "batch_size": 2048,
            "steps": steps,
            "components_optimizer": {
                "lr_schedule": COSINE(2e-3), "betas": [0.9, 0.999],
                "weight_decay": 0.0, "grad_clip_norm": 1.0,
            },
            "ci_fn_optimizer": {
                "lr_schedule": COSINE(2e-3), "betas": [0.9, 0.999],
                "weight_decay": 0.0, "grad_clip_norm": 1.0,
            },
            "faithfulness_warmup_steps": 200,
            "faithfulness_warmup_lr": 0.01,
            "faithfulness_warmup_weight_decay": 0.0,
            "loss_metrics": [
                {
                    "type": "ImportanceMinimalityLoss",
                    "coeff": imp_coeff,
                    "gamma": {
                        "max_val": 1.0,
                        "points": [
                            {"at": 0.0, "frac": 1.0},
                            {"at": 0.9, "frac": 0.01},
                            {"at": 1.0, "frac": 0.01},
                        ],
                    },
                    "frequency": {
                        "coeff": imp_coeff / 3.0,
                        "reference_datapoint_count": 65536,
                        "ema_halflife_steps": 200,
                    },
                },
                {
                    "type": "NonlinearityLocalityLoss",
                    "coeff": 3.0e-05,
                    "relative_threshold": {
                        "max_val": 8.0,
                        "points": [{"at": 0.0, "frac": 1.0}, {"at": 1.0, "frac": 0.25}],
                    },
                    "unit_kind_coefficients": {"neuron": 1.0},
                },
                {
                    "type": "MergedStochasticSubsetPPGDReconLoss",
                    "coeff": 1.0,
                    "adv_fraction": 0.5,
                    "n_warmup_steps": 2,
                    "routing": {"type": "uniform_k_subset"},
                    "source_shape": "bc",
                    "optimizer": {
                        "type": "adam", "beta1": 0.01, "beta2": 0.99, "eps": 1.0e-08,
                        "lr_schedule": {
                            "max_val": 0.01,
                            "points": [
                                {"at": 0.0, "frac": 0.0},
                                {"at": 0.025, "frac": 1.0},
                                {"at": 1.0, "frac": 1.0},
                            ],
                        },
                    },
                },
                {"type": "FaithfulnessLoss", "coeff": 1000.0},
            ],
        },
        "eval": {
            "batch_size": 2048,
            "n_steps": 1,
            "every": 1000,
            "slow_every": 5000,
            "slow_on_first_step": True,
            "metrics": [
                {"type": "PGDReconLoss", "coeff": None, "init": "random",
                 "source_shape": "c", "n_steps": 20, "step_size": 0.1},
                {"type": "CI_L0", "groups": {"total": ["*"]}, "ci_alive_threshold": 0.0},
            ],
        },
        "wandb": {
            "project": "param-decomp",
            "group": "memory-toy-vpd",
            "tags": [arch, f"k{k}", f"imp{imp_coeff:g}", f"ci-{ci}"]
                    + ([f"C{c_all}"] if c_all else []),
        },
        "target": {
            "arch": arch,
            "weights_path": f"/data/memory-toy/runs/f2e{k}{suffix}/model_final.safetensors",
            "expected_acc": expected_acc,
        },
        "data": {"facts_path": FACTS_PATH, "n_facts": 2**k},
    }


def stable_run_id(run_key: str) -> str:
    return "p-" + hashlib.md5(run_key.encode()).hexdigest()[:8]


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("wandb")])
def vpd(run_key: str, config: dict, facts_bytes: bytes) -> dict:
    import sys
    from pathlib import Path as P_

    facts_file = P_(FACTS_PATH)
    if not facts_file.exists():
        facts_file.parent.mkdir(parents=True, exist_ok=True)
        facts_file.write_bytes(facts_bytes)
        vol.commit()

    import yaml

    cfg_path = P_("/tmp/config.yaml")
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
    sys.path.insert(0, "/root/vpd")
    from run_memory_toy import main

    run_id = stable_run_id(run_key)
    try:
        main(str(cfg_path), data_root=P_(DATA_ROOT), run_id=run_id)
    except AssertionError as err:
        if "nothing left to run" not in str(err):
            raise  # already fully trained -> fall through to the summary
    finally:
        vol.commit()

    # summarize: last fast-eval + last train records from metrics.jsonl
    metrics_path = P_(DATA_ROOT) / "runs" / run_id / "metrics.jsonl"
    records = []
    if metrics_path.exists():
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    summary = {"run_key": run_key, "run_id": run_id, "n_records": len(records)}
    for rec in records:
        keep = {
            key: value
            for key, value in rec.items()
            if isinstance(value, (int, float))
            and any(tag in key for tag in ("PGDRecon", "l0", "L0", "Faithfulness",
                                           "ImportanceMinimality", "total_loss", "step"))
        }
        summary.update(keep)  # later records overwrite: ends at the final values
    return summary


PILOT_COEFFS = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)


@app.local_entrypoint()
def main(mode: str = "smoke", coeff: float = 1e-4, cells: str = "", steps: int = 30_000):
    here = Path(__file__).resolve().parent
    facts_bytes = (here / "cache" / "facts_seed0.npz").read_bytes()

    def expected_acc(arch: str, k: int) -> float:
        name = {"attn": "sweep_summary.json", "twoemb": "sweep_summary_twoemb.json",
                "mix": "sweep_summary_mix.json"}[arch]
        rows = json.loads((here / "cache" / name).read_text())
        return next(r["acc"] for r in rows if r["k"] == k)

    if mode == "smoke":
        cfg = build_config("attn", 15, 1e-4, expected_acc("attn", 15),
                           steps=600, save_every=300)
        cfg["eval"].update(every=200, slow_every=400, slow_on_first_step=True)
        print(vpd.remote("smoke-attn-k15", cfg, facts_bytes))
        return

    if mode == "pilot":
        jobs = [
            (f"pilot-attn-k16-imp{c:g}",
             build_config("attn", 16, c, expected_acc("attn", 16)), facts_bytes)
            for c in PILOT_COEFFS
        ]
    elif mode == "final":
        jobs = [
            (f"final-{arch}-k{k}-imp{coeff:g}",
             build_config(arch, k, coeff, expected_acc(arch, k)), facts_bytes)
            for arch in ("attn", "twoemb", "mix")
            for k in (15, 16, 17)
        ]
    elif mode == "lookup":  # lookup-table CI diagnostics (see lookup_ci.py)
        # k15 cells, same coeff as each cell's chosen MLP-CI run (user decision 2026-09-19);
        # v3 (user request): C = 1600 for EVERY matrix (v2 = original Cs, bf16-safe tap)
        diag = [("attn", 15, 1e-4), ("twoemb", 15, 1e-3), ("mix", 15, 1e-4)]
        jobs = [
            (f"lookup3-{arch}-k{k}-imp{c:g}",
             build_config(arch, k, c, expected_acc(arch, k), ci="lookup", c_all=1600),
             facts_bytes)
            for arch, k, c in diag
        ]
    elif mode == "lookup-sweep":  # coeff re-sweep for the C=1600 lookup attn/mix cells
        jobs = [
            (f"lookup3-{arch}-k15-imp{c:g}",
             build_config(arch, 15, c, expected_acc(arch, 15), ci="lookup", c_all=1600),
             facts_bytes)
            for arch in ("attn", "mix")
            for c in (3e-5, 1e-5, 3e-6)
        ]
    elif mode == "lookup-big":  # C > n_facts: k12 targets (4,096 facts), C = 8,192/matrix
        # coeff grid spans the plausible range: imp-min pressure scales ~ coeff * C, and
        # C is 10-40x the earlier healthy configs, so the knee should sit well below 1e-4
        jobs = [
            (f"lookup4-{arch}-k12-imp{c:g}-C8192",
             build_config(arch, 12, c, expected_acc(arch, 12), ci="lookup", c_all=8192),
             facts_bytes)
            for arch in ("attn", "twoemb", "mix")
            for c in (1e-4, 1e-5, 1e-6)
        ]
    elif mode == "cells":  # e.g. --cells attn:15,mix:17 --coeff 1e-4
        parsed = [c.split(":") for c in cells.split(",") if c]
        jobs = [
            (f"final-{arch}-k{k}-imp{coeff:g}-s{steps}" if steps != 30_000
             else f"final-{arch}-k{k}-imp{coeff:g}",
             build_config(arch, int(k), coeff, expected_acc(arch, int(k)), steps=steps),
             facts_bytes)
            for arch, k in parsed
        ]
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    results = list(vpd.starmap(jobs))
    out = here / "cache" / f"vpd_{mode}_summary.json"
    out.write_text(json.dumps(results, indent=2))
    for r in results:
        print(json.dumps(r))
