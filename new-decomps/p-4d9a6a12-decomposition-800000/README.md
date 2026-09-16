# p-4d9a6a12 decomposition (step 800000)

This bundle contains the trained decomposition only: the rank-1 component factors
(`U` and `V`) and the causal-importance function. Optimizer, adversarial-source,
and other training state are deliberately omitted. The small frozen target checkpoint
needed to reconstruct the public loader's array shapes is included under
`pretrain_cache/`.

It was verified against public `goodfire-ai/param-decomp` commit
`facf2e7b1d5273985ea1af277240f340f5bdeac0` on CPU:

```python
from pathlib import Path
from param_decomp.experiments.lm.load_run import open_jax_run

root = Path("p-4d9a6a12-decomposition-800000")
loaded = open_jax_run(
    root / "runs" / "p-4d9a6a12",
    step=800000,
    data_root=root,
)
components = loaded.prepared_weights  # {matrix_kind: {"U": ..., "V": ...}}
causal_importance = loaded.ci_fn
```

Install the pinned public source with:

```bash
pip install "git+https://github.com/goodfire-ai/param-decomp.git@facf2e7b1d5273985ea1af277240f340f5bdeac0"
```

`launch_config.yaml` is the original training config. `deliverable.yaml` is a
consumer-only normalization for that public commit: it removes the initialization-only
field (irrelevant after training) and makes the default seed explicit. The checkpoint
payload itself is unchanged from the final internal artifact.

Source run: https://wandb.ai/goodfire/param-decomp/runs/p-4d9a6a12

Crew-Address: agent/msia
