"""Modal reference forward pass for the sink target models: JAX logits/loss on
the first N cached Pile rows, to verify the local Torch port (model_def.py with
attention_sinks/untied head + load.load_sink) numerically.

Runs the target via the sink-loader PR commit's open_jax_run (decomposition C's
target = t-87f91319 = "sink seed 45"), returns per-row mean NLL and float16
logits at a stride of positions of row 0 over the wire; the local comparison is
done by verify_torch_port_local.py.

Run:  modal run sink-models/hide/verify_torch_port_modal.py
"""

import modal

app = modal.App("verify-sink-torch-port")
vol = modal.Volume.from_name("vpd-4layer")

COMMIT = "82a67f71c21d9cc9832f2dfced69c4f03d68e992"  # PR #1002 sink loader
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"param-decomp[cuda] @ git+https://github.com/goodfire-ai/param-decomp.git@{COMMIT}"
    )
)

N_ROWS = 8
STRIDE = 16  # keep logits at positions 0, 16, 32, ... of row 0


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=1200)
def reference() -> dict:
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_default_matmul_precision", "highest")
    from param_decomp.experiments.lm.load_run import open_jax_run

    root = Path("/data/sink-models")
    loaded = open_jax_run(root / "runs" / "p-d60af588", step=100000, data_root=root)
    rows = np.load("/data/pile_rows.npy")[:N_ROWS]  # (N, 512) int32

    with jax.set_mesh(loaded.mesh):
        out = loaded.placed.clean_forward(jnp.asarray(rows), ())
    logits = np.asarray(out.output, dtype=np.float32)  # (N, 512, vocab)

    logp = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    tgt = rows[:, 1:]
    nll = -np.take_along_axis(np.asarray(logp)[:, :-1], tgt[..., None], axis=-1)[..., 0]
    return {
        "mean_nll": float(nll.mean()),
        "row_nll": nll.mean(axis=1).astype(np.float64),
        "logits_row0": logits[0, ::STRIDE].astype(np.float16),  # (32, vocab)
        "stride": STRIDE,
    }


@app.local_entrypoint()
def main() -> None:
    import numpy as np

    out = reference.remote()
    print("JAX reference mean NLL:", out["mean_nll"])
    print("per-row NLL:", np.round(out["row_nll"], 4))
    np.savez(
        "sink-models/hide/cache/jax_reference.npz",
        mean_nll=out["mean_nll"],
        row_nll=out["row_nll"],
        logits_row0=out["logits_row0"],
        stride=out["stride"],
        n_rows=N_ROWS,
    )
    print("saved sink-models/hide/cache/jax_reference.npz")
