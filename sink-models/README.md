# Task 1423: untied-head, attention-sink model and decomposition bundles

These files are independently extractable into one data root. Download one decomposition
archive and its target base archive, then extract both in the same directory.

| decomposition | target base archive |
|---|---|
| `p-d60af588-decomposition-step100000.tar.zst` | `t-87f91319-base-step100000.tar.zst` |
| `p-fecd6a6b-decomposition-step100000.tar.zst` | `t-87f91319-base-step100000.tar.zst` |
| `p-bd411e35-decomposition-step100000.tar.zst` | `t-75f6c439-base-step100000.tar.zst` |

The public attention-sink loader is at https://github.com/goodfire-ai/param-decomp/pull/1002
(commit `82a67f71c`). Until it merges, clone the public repository and check out the branch:

```bash
git clone https://github.com/goodfire-ai/param-decomp.git
cd param-decomp
git checkout bridge/task-1423-external-loader
uv sync --all-packages --extra cuda
```

Extract and restore a decomposition:

```bash
mkdir task1423-data
cd task1423-data
tar --zstd -xf ../p-d60af588-decomposition-step100000.tar.zst
tar --zstd -xf ../t-87f91319-base-step100000.tar.zst
cd ../param-decomp
uv run python - <<'PY'
from pathlib import Path
from param_decomp.experiments.lm.load_run import (
    SINGLE_DEVICE_RESIDENT_LAYOUT,
    open_jax_run,
)
root = Path("../task1423-data").resolve()
loaded = open_jax_run(
    root / "runs/p-d60af588",
    100000,
    data_root=root,
    layout=SINGLE_DEVICE_RESIDENT_LAYOUT,
)
print(loaded.run_id, loaded.step, type(loaded.model).__name__)
PY
```

Each decomposition archive contains only the read-only Orbax `decomposition` item, its
checkpoint metadata, and its product configs. It deliberately excludes the 51 GB training
state. Each base archive contains the exact model config and safetensors weights used by the
loader. `SHA256SUMS` covers every downloadable archive.

The activating-example harvest bundles use the same extraction convention and will be added
when the three 10M-token harvest jobs finish.

Crew-Address: agent/i9y2
