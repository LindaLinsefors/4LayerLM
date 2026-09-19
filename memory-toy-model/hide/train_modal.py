"""Train the 1-layer memory toy models on Modal (JAX, full-batch AdamW).

Three architectures (decided with Linda 2026-09-18), all: vocab 1024,
d_model 96, MLP 384 (NewGELU), RMSNorm, no biases, UNTIED lm_head, NO
positional encoding, loss only on predicting t2 from (t0, t1):

  arch="attn"   1 layer, 3 heads x head_dim 32. Self is masked -> position 1
                sees only key 0 plus a learned per-head sink logit (GPT-OSS
                zero-value slot), so each head is a sigmoid gate
                a_h = sigmoid(q_h(t1).k_h(t0)/sqrt(32) - s_h) on the t0 value.
  arch="twoemb" NO attention. A second full embedding matrix for position 0:
                h = wte0[t0] + wte[t1], then the MLP block as usual.
  arch="mix"    NO attention, shared embedding, learned 96x96 mixing matrix
                on the position-0 vector: h = W_mix wte[t0] + wte[t1].

Task: memorize n = 2^k random facts (t0,t1) -> t2 (nested prefixes of
hide/cache/facts_seed0.npz). One Modal container per k, full-batch AdamW
(weight decay 0.1 on ndim>=2 params, matching the model family), cosine LR,
early stop after the whole dataset stays memorized for STOP_EVALS evals.

Outputs per run on volume vpd-4layer under /data/memory-toy/runs/f2e<k>/
("attn"; the other archs append -twoemb / -mix):
  model_final.safetensors   (family-style key names: h.0.attn.q_proj.weight, ...)
  metrics.json              (eval trajectory: step, loss, acc, nll)
  config.json               (dims, seeds, optimizer, dataset spec)
Summaries also returned to the local entrypoint -> hide/cache/sweep_summary.json.

Run (PowerShell, needs PYTHONUTF8=1 for modal on this machine):
  modal run memory-toy-model/hide/train_modal.py --smoke --archs attn   # k=11, quick check
  modal run --detach memory-toy-model/hide/train_modal.py --archs attn,twoemb,mix
"""

import modal

app = modal.App("memory-toy-train")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "jax[cuda12]==0.6.2", "optax==0.2.4", "numpy", "safetensors", "wandb"
)
vol = modal.Volume.from_name("vpd-4layer")

V, D, H, HD, M = 1024, 96, 3, 32, 384
MAX_STEPS = 50_000
EVAL_EVERY = 250
STOP_EVALS = 4  # stop after acc == 1.0 this many consecutive evals
PEAK_LR, END_LR, WARMUP = 3e-3, 3e-5, 200
WEIGHT_DECAY = 0.1
INIT_SEED = 0


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=4 * 3600,
              secrets=[modal.Secret.from_name("wandb")])
def train(k: int, facts_bytes: bytes, arch: str = "attn", max_steps: int = MAX_STEPS) -> dict:
    import io
    import json
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from safetensors.numpy import save_file

    facts = np.load(io.BytesIO(facts_bytes))["facts"][: 2**k].astype(np.int32)
    t0, t1, t2 = (jnp.asarray(facts[:, i]) for i in range(3))

    ks = jax.random.split(jax.random.PRNGKey(INIT_SEED), 9)
    init = lambda kk, shape: 0.02 * jax.random.normal(kk, shape, jnp.float32)
    params = {
        "wte": init(ks[0], (V, D)),
        "fc": init(ks[5], (M, D)),  # nn.Linear orientation: (out, in)
        "down": init(ks[6], (D, M)),
        "lm_head": init(ks[7], (V, D)),
        "g2": jnp.ones(D),
        "gf": jnp.ones(D),
    }
    if arch == "attn":
        params |= {
            "q": init(ks[1], (D, D)), "k": init(ks[2], (D, D)),
            "v": init(ks[3], (D, D)), "o": init(ks[4], (D, D)),
            "g1": jnp.ones(D), "sinks": jnp.zeros(H),
        }
    elif arch == "twoemb":
        params["wte0"] = init(ks[8], (V, D))
    elif arch == "mix":
        params["mix"] = init(ks[8], (D, D))
    else:
        raise ValueError(arch)

    def rms(x, g):
        return g * x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)

    def fwd(p, t0, t1):
        x1 = p["wte"][t1]
        if arch == "attn":
            x0 = p["wte"][t0]
            n0, n1 = rms(x0, p["g1"]), rms(x1, p["g1"])
            q = (n1 @ p["q"].T).reshape(-1, H, HD)
            kk = (n0 @ p["k"].T).reshape(-1, H, HD)
            vv = (n0 @ p["v"].T).reshape(-1, H, HD)
            score = jnp.einsum("bhd,bhd->bh", q, kk) / jnp.sqrt(HD)
            a = jax.nn.sigmoid(score - p["sinks"])  # softmax over {key 0, sink}
            h = x1 + (a[..., None] * vv).reshape(-1, D) @ p["o"].T
        elif arch == "twoemb":
            h = p["wte0"][t0] + x1
        else:  # mix
            h = p["wte"][t0] @ p["mix"].T + x1
        h = h + jax.nn.gelu(rms(h, p["g2"]) @ p["fc"].T, approximate=True) @ p["down"].T
        return rms(h, p["gf"]) @ p["lm_head"].T

    def loss_fn(p):
        lg = fwd(p, t0, t1)
        return optax.softmax_cross_entropy_with_integer_labels(lg, t2).mean()

    sched = optax.warmup_cosine_decay_schedule(0.0, PEAK_LR, WARMUP, max_steps, END_LR)
    opt = optax.adamw(
        sched, b1=0.9, b2=0.95, weight_decay=WEIGHT_DECAY,
        mask=jax.tree.map(lambda x: x.ndim >= 2, params),
    )
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    @jax.jit
    def evaluate(p):
        lg = fwd(p, t0, t1)
        nll = optax.softmax_cross_entropy_with_integer_labels(lg, t2)
        return (jnp.argmax(lg, -1) == t2).mean(), nll.mean()

    import wandb

    wandb.init(project="param-decomp", group="memory-toy-pretrain",
               name=f"memtoy-{arch}-f2e{k}", config={"arch": arch, "k": k},
               tags=[arch, f"k{k}"])

    metrics, perfect_streak, t_start = [], 0, time.time()
    n_steps = 0
    for i in range(max_steps):
        params, opt_state, loss = step(params, opt_state)
        n_steps = i + 1
        if n_steps % EVAL_EVERY == 0 or n_steps == max_steps:
            acc, nll = evaluate(params)
            acc, nll = float(acc), float(nll)
            metrics.append({"step": n_steps, "loss": float(loss), "acc": acc, "nll": nll})
            wandb.log(metrics[-1], step=n_steps)
            perfect_streak = perfect_streak + 1 if acc == 1.0 else 0
            if perfect_streak >= STOP_EVALS:
                break

    acc, nll = evaluate(params)
    summary = {
        "arch": arch, "k": k, "n_facts": int(2**k), "steps": n_steps,
        "acc": float(acc), "nll": float(nll),
        "n_wrong": int(round((1 - float(acc)) * 2**k)),
        "minutes": round((time.time() - t_start) / 60, 1),
    }

    suffix = "" if arch == "attn" else f"-{arch}"
    run_dir = Path(f"/data/memory-toy/runs/f2e{k}{suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {  # family-style names so later VPD tooling feels at home
        "wte.weight": params["wte"],
        "h.0.mlp.c_fc.weight": params["fc"],
        "h.0.mlp.down_proj.weight": params["down"],
        "h.0.rms_2.weight": params["g2"],
        "ln_f.weight": params["gf"],
        "lm_head.weight": params["lm_head"],
    }
    if arch == "attn":
        state |= {
            "h.0.attn.q_proj.weight": params["q"],
            "h.0.attn.k_proj.weight": params["k"],
            "h.0.attn.v_proj.weight": params["v"],
            "h.0.attn.o_proj.weight": params["o"],
            "h.0.attn.sinks": params["sinks"],
            "h.0.rms_1.weight": params["g1"],
        }
    elif arch == "twoemb":
        state["wte0.weight"] = params["wte0"]
    else:
        state["mix.weight"] = params["mix"]
    wandb.summary.update(summary)
    wandb.finish()
    save_file({n: np.asarray(v) for n, v in state.items()}, run_dir / "model_final.safetensors")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "config.json").write_text(json.dumps({
        "arch": arch,
        "vocab": V, "d_model": D, "d_mlp": M, "n_layers": 1,
        **({"n_heads": H, "head_dim": HD, "self_attention": "masked",
            "attention_sinks": True} if arch == "attn" else
           {"attention": "none",
            "pair_combine": "wte0[t0] + wte[t1]" if arch == "twoemb"
                            else "mix @ wte[t0] + wte[t1]"}),
        "positional_encoding": "none", "tie_word_embeddings": False,
        "activation": "gelu_tanh", "norm": "rmsnorm(eps=1e-6)", "biases": False,
        "dataset": {"file": "facts_seed0.npz", "seed": 0, "nested_prefix": int(2**k)},
        "init_seed": INIT_SEED, "optimizer": "adamw(full batch)",
        "peak_lr": PEAK_LR, "end_lr": END_LR, "warmup": WARMUP,
        "b1": 0.9, "b2": 0.95, "weight_decay": WEIGHT_DECAY,
        "decay_mask": "ndim>=2", "max_steps": max_steps,
        "eval_every": EVAL_EVERY, "stop_evals": STOP_EVALS,
        "loss": "CE on t2 only", "summary": summary,
    }, indent=2))
    vol.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False, archs: str = "attn"):
    import json
    from pathlib import Path

    here = Path(__file__).resolve().parent
    facts_bytes = (here / "cache" / "facts_seed0.npz").read_bytes()
    arch_list = archs.split(",")

    if smoke:
        for arch in arch_list:
            print(train.remote(11, facts_bytes, arch, max_steps=6000))
        return

    jobs = [(k, facts_bytes, arch) for arch in arch_list for k in range(11, 19)]
    results = list(train.starmap(jobs))
    for arch in arch_list:
        out = here / "cache" / (f"sweep_summary.json" if arch == "attn"
                                else f"sweep_summary_{arch}.json")
        out.write_text(json.dumps([r for r in results if r["arch"] == arch], indent=2))
    print(f"{'arch':>7} {'k':>3} {'n_facts':>8} {'steps':>7} {'acc':>8} {'nll':>9} {'wrong':>6} {'min':>5}")
    for r in results:
        print(f"{r['arch']:>7} {r['k']:>3} {r['n_facts']:>8} {r['steps']:>7} {r['acc']:>8.4f} "
              f"{r['nll']:>9.4f} {r['n_wrong']:>6} {r['minutes']:>5}")
