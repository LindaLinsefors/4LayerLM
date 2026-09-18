# Task 1423: untied-head, attention-sink models, decompositions, and harvests

These files are independently extractable into one data root. Download one decomposition
archive, its target base archive, and optionally its activation/example harvest archive, then
extract them in the same directory.

| decomposition | target base archive | activation/example harvest |
|---|---|---|
| `p-d60af588-decomposition-step100000.tar.zst` | `t-87f91319-base-step100000.tar.zst` | `p-d60af588-harvest-10m-step100000.tar.zst` |
| `p-fecd6a6b-decomposition-step100000.tar.zst` | `t-87f91319-base-step100000.tar.zst` | `p-fecd6a6b-harvest-10m-step100000.tar.zst` |
| `p-bd411e35-decomposition-step100000.tar.zst` | `t-75f6c439-base-step100000.tar.zst` | `p-bd411e35-harvest-10m-step100000.tar.zst` |

Files are served without a W&B login from:

```text
https://api.wandb.ai/files/goodfire/param-decomp/share-task-1423/<filename>
```

The public attention-sink loader is at https://github.com/goodfire-ai/param-decomp/pull/1002
(commit `82a67f71c`). Until it merges, clone the public repository and check out the branch:

```bash
git clone https://github.com/goodfire-ai/param-decomp.git
cd param-decomp
git checkout bridge/task-1423-external-loader
uv sync --all-packages --extra cuda
```

Extract and restore a decomposition and its harvest:

```bash
mkdir task1423-data
cd task1423-data
tar --zstd -xf ../p-d60af588-decomposition-step100000.tar.zst
tar --zstd -xf ../t-87f91319-base-step100000.tar.zst
tar --zstd -xf ../p-d60af588-harvest-10m-step100000.tar.zst
cd ../param-decomp
uv run python - <<'PY'
from pathlib import Path
from param_decomp.experiments.lm.load_run import SINGLE_DEVICE_RESIDENT_LAYOUT, open_jax_run
from param_decomp.harvest.domain import CheckpointRef, HarvestRef
from param_decomp.harvest.reader import HarvestReader

root = Path("../task1423-data").resolve()
loaded = open_jax_run(
    root / "runs/p-d60af588", 100000, data_root=root, layout=SINGLE_DEVICE_RESIDENT_LAYOUT
)
print(loaded.run_id, loaded.step, type(loaded.model).__name__)

reference = HarvestRef(
    checkpoint=CheckpointRef(
        run_id="p-d60af588", checkpoint_id="step-100000", step=100000
    ),
    harvest_id="h-p-d60af588-s100000-10m-rep100-task1423-v1",
)
with HarvestReader.open(root, reference) as harvest:
    key = harvest.observed_component_keys()[0]
    component = harvest.component(key)
    print(key, component.examples.count, component.examples.activations.shape)
PY
```

Each decomposition archive contains only the read-only Orbax `decomposition` item, its
checkpoint metadata, and product configs; it deliberately excludes the 51 GB training state.
Each base archive contains the exact model config and safetensors weights used by the loader.

Each harvest scanned 2,442 batches of 8 x 512 tokens (10,002,432 corpus tokens), covering all
24 decomposition sites and 36,864 components. It stores up to 100 representative 41-token
windows per component where lower-leaky causal importance exceeds 0.1, including the firing
trace, causal importance, and normalized activation trace. Rare or inactive components can
have fewer than 100 examples; the manifest and SQLite index preserve full coverage statistics.
`SHA256SUMS` covers every downloadable archive.

Crew-Address: agent/i9y2
