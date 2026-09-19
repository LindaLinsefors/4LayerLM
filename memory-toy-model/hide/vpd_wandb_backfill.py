"""Backfill wandb from the memory-toy runs already on volume vpd-4layer.

Replays (a) every toy pretraining run's metrics.json (/memory-toy/runs/f2e*) and
(b) every VPD run's metrics.jsonl (/memory-toy/vpd/runs/p-*) into wandb project
param-decomp (groups memory-toy-pretrain / memory-toy-vpd, tag "backfill"). VPD run
names get the run id suffixed (the yaml run_name is shared between MLP- and lookup-CI
variants). Idempotent: backfilled dirs are recorded in /memory-toy/wandb_backfilled.json
on the volume, so re-running only picks up NEW runs (e.g. after a sweep finishes).

Run (PYTHONUTF8=1):  modal run memory-toy-model/hide/vpd_wandb_backfill.py
"""

import json
from pathlib import Path

import modal

app = modal.App("memory-toy-wandb-backfill")
image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb", "pyyaml")
vol = modal.Volume.from_name("vpd-4layer")
MARKER = Path("/data/memory-toy/wandb_backfilled.json")


@app.function(image=image, volumes={"/data": vol}, timeout=3600,
              secrets=[modal.Secret.from_name("wandb")])
def backfill() -> list[str]:
    import wandb
    import yaml

    done: set[str] = set(json.loads(MARKER.read_text())) if MARKER.exists() else set()
    newly: list[str] = []

    def numeric(rec: dict) -> dict:
        return {key: v for key, v in rec.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}

    # (a) toy pretraining runs
    for run_dir in sorted(Path("/data/memory-toy/runs").glob("f2e*")):
        tag = f"pretrain:{run_dir.name}"
        metrics_path = run_dir / "metrics.json"
        if tag in done or not metrics_path.exists():
            continue
        config = json.loads((run_dir / "config.json").read_text())
        summary = config.get("summary", {})
        arch = config.get("arch", "attn")
        k = config["dataset"]["nested_prefix"].bit_length() - 1
        run = wandb.init(project="param-decomp", group="memory-toy-pretrain",
                         name=f"memtoy-{arch}-f2e{k}", config=config,
                         tags=[arch, f"k{k}", "backfill"], reinit=True)
        for rec in json.loads(metrics_path.read_text()):
            run.log(numeric(rec), step=rec["step"])
        run.summary.update(summary)
        run.finish()
        done.add(tag)
        newly.append(tag)

    # (b) VPD decomposition runs
    for run_dir in sorted(Path("/data/memory-toy/vpd/runs").glob("p-*")):
        tag = f"vpd:{run_dir.name}"
        metrics_path = run_dir / "metrics.jsonl"
        if tag in done or not metrics_path.exists():
            continue
        cfg = yaml.safe_load((run_dir / "launch_config.yaml").read_text())
        if cfg.get("wandb"):  # run logs live -- a backfill copy would be a duplicate
            done.add(tag)
            continue
        ci_type = cfg["decomposition"]["ci"]["type"]
        cs = sorted({s["C"] for s in cfg["decomposition"]["sites"]["sites"]})
        arch = cfg["target"]["arch"]
        k = int(cfg["data"]["n_facts"]).bit_length() - 1
        run = wandb.init(
            project="param-decomp", group="memory-toy-vpd",
            name=f"{cfg['run_name']}--{run_dir.name}", config=cfg,
            tags=[arch, f"k{k}", f"ci-{ci_type}", f"C{'-'.join(map(str, cs))}", "backfill"],
            reinit=True,
        )
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            step = rec.get("step")
            if step is not None:
                run.log(numeric(rec), step=int(step))
        run.finish()
        done.add(tag)
        newly.append(tag)

    MARKER.write_text(json.dumps(sorted(done)))
    vol.commit()
    return newly


@app.local_entrypoint()
def main():
    for tag in backfill.remote():
        print("backfilled", tag)


@app.function(image=image, volumes={"/data": vol}, timeout=1800,
              secrets=[modal.Secret.from_name("wandb")])
def repair(run_ids: str) -> list[str]:
    """Delete partial backfill wandb runs for the given run dirs (comma-separated ids),
    drop their marker entries, and re-backfill them from the now-complete metrics."""
    import wandb

    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    api = wandb.Api()
    entity = api.default_entity
    deleted = []
    for run in api.runs(f"{entity}/param-decomp", filters={"tags": "backfill"}):
        if any(run.name.endswith(f"--{rid}") for rid in ids):
            run.delete()
            deleted.append(run.name)
    done = set(json.loads(MARKER.read_text())) if MARKER.exists() else set()
    done -= {f"vpd:{rid}" for rid in ids}
    MARKER.write_text(json.dumps(sorted(done)))
    vol.commit()
    return deleted


@app.local_entrypoint(name="repair-main")
def repair_main(run_ids: str):
    for name in repair.remote(run_ids):
        print("deleted partial:", name)
    for tag in backfill.remote():
        print("re-backfilled", tag)
