"""The memory-toy `DecomposedModel` for param-decomp (JAX, latest main 5ffc73d79).

Three 1-layer memorization targets trained in memory-toy-model/hide/train_modal.py
(vocab 1024, d_model 96, MLP 384, NewGELU tanh, RMSNorm eps 1e-6, no biases, untied
lm_head, no positional encoding), differing in how token 0 reaches the MLP:

  arch="attn"    3 self-masked sink-gated heads: a_h = sigmoid(q_h(t1).k_h(t0)/sqrt(32)
                 - s_h); h = wte[t1] + o_proj(gate * v(t0)).
                 Sites: q_proj k_proj v_proj o_proj c_fc down_proj.
  arch="twoemb"  h = wte0[t0] + wte[t1].          Sites: c_fc down_proj.
  arch="mix"     h = mix @ wte[t0] + wte[t1].     Sites: mix c_fc down_proj.

Only position 1's prediction exists, so the model is positionless: input = token pairs
[B, 2] int32, output = logits [B, vocab]. Recon loss = KL on the logits
(targets.losses.kl_per_position). Embeddings / lm_head / norms / sinks stay frozen and
undecomposed (Linears-only decision, 2026-09-18). Modeled closely on targets/resid_mlp.py.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteCI,
    SiteDims,
    SiteSpec,
    component_stacks_from_site_arrays,
    require_full_emission,
    site_slots_for,
)
from param_decomp.core.decomposed_linear import site_out
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
)
from param_decomp.core.nonlinearity import Neurons
from param_decomp.core.placement import PlacementRules
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.targets.glu_transformer import _neuron_aligned_site_factors
from param_decomp.targets.linear_site_capture import (
    SiteCaptureGrammar,
    SiteCaptureSources,
    capture_keys_by_site,
    record_requested_value,
    site_output_key,
)
from param_decomp.targets.losses import kl_per_position

VOCAB, D, H, HD, M = 1024, 96, 3, 32, 384
RMS_EPS = 1e-6

FACT_IDX_KEY = "fact_idx"
"""Special capture key: the fact's row index in the training table (tokens column 2,
present when the sampler emits (B,3) batches), recorded as a (B,2) tap holding the
base-256 digits (idx//256, idx%256) — each < 256 so the value survives the engine's
bfloat16 compute cast exactly. Not a site capture — filtered out before the grammar."""

ARCH_SITES: dict[str, tuple[str, ...]] = {
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"),
    "twoemb": ("c_fc", "down_proj"),
    "mix": ("mix", "c_fc", "down_proj"),
}

# neuron-aligned init: which side of W carries the architectural units
# (heads/neurons live on q/k/v outputs and o/down inputs, as in the LM initializer)
_UNITS_ON_INPUT = {"o_proj": True, "down_proj": True}


@dataclass(frozen=True)
class MemoryToyRunTarget:
    """Carried on `BuiltRun.target` (satisfies the core `TargetSites` protocol)."""

    arch: str
    weights_path: str
    expected_acc: float
    facts_path: str
    n_facts: int
    global_batch: int
    sites: tuple[SiteC, ...]


class MemoryToyTarget(eqx.Module):
    """Frozen memory-toy weights. Per-arch optional fields are `None` when absent."""

    arch: str = eqx.field(static=True)
    wte: Float[Array, "vocab d"]
    lm_head: Float[Array, "vocab d"]
    g2: Float[Array, " d"]
    gf: Float[Array, " d"]
    q: Float[Array, "d d"] | None
    k: Float[Array, "d d"] | None
    v: Float[Array, "d d"] | None
    o: Float[Array, "d d"] | None
    g1: Float[Array, " d"] | None
    sinks: Float[Array, " h"] | None
    wte0: Float[Array, "vocab d"] | None
    mix: Float[Array, "d d"] | None
    fc: Float[Array, "m d"]
    down: Float[Array, "d m"]


def _rms(x: Array, g: Array) -> Array:
    return g * x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + RMS_EPS)


def _embed(table: Array, ids: Array) -> Array:
    """Row gather with the output sharding spelled explicitly (jax 0.11 explicit-sharding
    refuses an ambiguous gather of a replicated table by batch-sharded ids)."""
    match jax.typeof(ids).sharding:
        case NamedSharding(mesh=mesh, spec=spec):
            batch_spec = spec[0] if len(spec) else None
            return table.at[ids].get(out_sharding=NamedSharding(mesh, P(batch_spec, None)))
        case other:
            raise AssertionError(f"expected a named sharding on token ids, got {type(other)}")


def canonical_site_cs(arch: str, site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    order = ARCH_SITES[arch]
    got = {s.name for s in site_cs}
    assert got == set(order), f"{arch} sites must be exactly {order}, got {sorted(got)}"
    by_name = {s.name: s for s in site_cs}
    return tuple(by_name[name] for name in order)


def site_dims(name: str) -> SiteDims:
    match name:
        case "q_proj" | "k_proj" | "v_proj" | "o_proj" | "mix":
            return SiteDims(d_in=D, d_out=D)
        case "c_fc":
            return SiteDims(d_in=D, d_out=M)
        case "down_proj":
            return SiteDims(d_in=M, d_out=D)
        case _:
            raise AssertionError(f"unknown memory-toy site {name!r}")


def site_specs(arch: str, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    site_cs = canonical_site_cs(arch, site_cs)
    return tuple(
        SiteSpec(
            name=site.name,
            factorization=site_dims(site.name).dense(site.C),
            group=site.name,  # independent per-site stacks (the toy pattern)
            nonlinearity_partition=Neurons() if site.name == "c_fc" else None,
        )
        for site in site_cs
    )


def _frozen_site_weight(target: MemoryToyTarget, name: str) -> Array:
    field = {
        "q_proj": target.q, "k_proj": target.k, "v_proj": target.v, "o_proj": target.o,
        "mix": target.mix, "c_fc": target.fc, "down_proj": target.down,
    }[name]
    assert field is not None, (target.arch, name)
    return field


def _model_forward(target, tokens: Int[Array, "B 2"], site_apply) -> Array:
    """The shared forward skeleton. `site_apply(name, W, x) -> y` is the one seam:
    frozen `x @ W.T` (clean) or the masked component sum (`site_out`)."""
    t0, t1 = tokens[:, 0], tokens[:, 1]
    x1 = _embed(target.wte, t1)
    if target.arch == "attn":
        n0 = _rms(_embed(target.wte, t0), target.g1)
        n1 = _rms(x1, target.g1)
        qh = site_apply("q_proj", target.q, n1).reshape(-1, H, HD)
        kh = site_apply("k_proj", target.k, n0).reshape(-1, H, HD)
        vh = site_apply("v_proj", target.v, n0).reshape(-1, H, HD)
        score = jnp.sum(qh * kh, -1) / jnp.sqrt(float(HD))
        gate = jax.nn.sigmoid(score - target.sinks)  # softmax over {key 0, sink}
        h = x1 + site_apply("o_proj", target.o, (gate[..., None] * vh).reshape(-1, D))
    elif target.arch == "twoemb":
        h = _embed(target.wte0, t0) + x1
    else:
        h = site_apply("mix", target.mix, _embed(target.wte, t0)) + x1
    pre = site_apply("c_fc", target.fc, _rms(h, target.g2))
    h = h + site_apply("down_proj", target.down, jax.nn.gelu(pre, approximate=True))
    return _rms(h, target.gf) @ target.lm_head.T


def clean_logits(target: MemoryToyTarget, tokens: Int[Array, "B 2"]) -> Array:
    return _model_forward(target, tokens, lambda _n, W, x: x @ W.T)


def _captured_forward(
    target: MemoryToyTarget,
    tokens: Array,
    requested_keys: tuple[str, ...],
    site_keys: tuple[str, ...],
    capture_sources: SiteCaptureSources,
    site_core,
) -> ForwardResult[Array]:
    """Run the forward with `site_core(name, W, x) -> y`, recording exactly the
    requested per-site input/output captures around each site (`site_keys` =
    `requested_keys` minus the special FACT_IDX_KEY tap)."""
    capture_keys = capture_keys_by_site(site_keys, capture_sources)
    captures: dict[str, Array] = {}
    if FACT_IDX_KEY in requested_keys:
        assert tokens.shape[-1] >= 3, "fact_idx capture needs (B,3) batches (t0, t1, idx)"
        # two base-256 digits, each bf16-EXACT: the engine casts CI inputs to bfloat16
        # (COMPUTE_DT), and a raw index > 256 would be quantized (integers are bf16-exact
        # only below 256 -- the 1,152-bucket bug of 2026-09-19)
        idx = tokens[:, 2]
        captures[FACT_IDX_KEY] = jnp.stack(
            [(idx // 256).astype(jnp.float32), (idx % 256).astype(jnp.float32)], axis=-1
        )

    def site_apply(name: str, W: Array, x: Array) -> Array:
        record_requested_value(captures, capture_keys.input_key_by_site.get(name), x)
        y = site_core(name, W, x)
        record_requested_value(captures, capture_keys.output_key_by_site.get(name), y)
        return y

    output = _model_forward(target, tokens, site_apply)
    assert set(captures) == set(requested_keys), (sorted(captures), sorted(requested_keys))
    return ForwardResult.from_producer(
        output=output,
        capture_keys=requested_keys,
        capture_values=tuple(captures[key] for key in requested_keys),
    )


class MemoryToyDecomposedModel(eqx.Module):
    """The memory-toy `DecomposedModel` (positionless; input = [B,2] token pairs,
    output = [B, vocab] logits, recon = KL on logits)."""

    target: MemoryToyTarget
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, placement: PlacementRules) -> "MemoryToyDecomposedModel":
        repl = NamedSharding(placement.mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Float[Array, ""]:
        return kl_per_position(masked_output, clean_output)

    @staticmethod
    def pin_output_batch(output: Array, mesh: Mesh | None) -> Array:
        return batch_shard_leading(output, mesh)

    def _capture_grammar(self) -> SiteCaptureGrammar:
        return SiteCaptureGrammar(sites=self.site_names, physical_source_of=lambda point: point)

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        assert set(sites) <= set(self.site_names), (sites, self.site_names)
        return tuple(site_output_key(site) for site in sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        self._capture_grammar().resolve(keys)

    def clean_forward(
        self,
        tokens: Int[Array, "B 2"],
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[Array]:
        del placement
        ordered = tuple(sorted(capture_keys))
        site_keys = tuple(k for k in ordered if k != FACT_IDX_KEY)
        sources = self._capture_grammar().resolve(site_keys)
        return _captured_forward(
            self.target, tokens, ordered, site_keys, sources, lambda _n, W, x: x @ W.T
        )

    def prepare_compute_weights(
        self, vu: ComponentStacks, placement: PlacementRules | None
    ) -> ComponentStacks:
        del placement
        return vu

    def component_activation_forward(
        self,
        prepared_weights: ComponentStacks,
        inputs: Array,
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[Array], dict[str, SiteCI]]:
        del prepared_weights, inputs, sites, capture_keys, placement
        raise NotImplementedError(
            f"{type(self).__name__} does not support component-activation harvest"
        )

    def stack_ci(self, ci_lower: Mapping[str, SiteCI]) -> dict[str, Array]:
        return {name: require_full_emission(value) for name, value in ci_lower.items()}

    def masked_forward(
        self,
        prepared_weights: ComponentStacks,
        tokens: Int[Array, "B 2"],
        /,
        *,
        masking: Masking,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[Array]:
        ordered = tuple(sorted(capture_keys))
        site_keys = tuple(k for k in ordered if k != FACT_IDX_KEY)
        sources = self._capture_grammar().resolve(site_keys)
        explicit = materialize_masking(masking)
        component_masks = {
            name: require_full_emission(m) for name, m in explicit.component_masks.items()
        }
        assert set(component_masks) == set(self.site_names), (
            sorted(component_masks), self.site_names,
        )

        def forward(
            vu: ComponentStacks,
            tokens: Array,
            component_masks: Mapping[str, Array],
            weight_delta_masks: Mapping[str, Array] | None,
            routes: Mapping[str, Bool[Array, "*leading"]] | None,
        ) -> ForwardResult[Array]:
            def site_core(name: str, W: Array, x: Array) -> Array:
                sc = vu.site(name)
                return site_out(
                    x,
                    sc.V,
                    sc.U,
                    W,
                    component_masks[name],
                    None if weight_delta_masks is None else weight_delta_masks[name],
                    None if routes is None else routes[name],
                    placement,
                    None,
                )

            return _captured_forward(self.target, tokens, ordered, site_keys, sources, site_core)

        forward = jax.checkpoint(forward) if remat else forward
        return forward(
            prepared_weights, tokens, component_masks,
            explicit.weight_delta_masks, explicit.routes,
        )

    def target_weight_sq_norms(self) -> dict[str, Array]:
        norms: dict[str, list[Array]] = {}
        for name, group, _slot in site_slots_for(self.sites):
            W = _frozen_site_weight(self.target, name)
            norms.setdefault(group, []).append(jnp.sum(W.astype(jnp.float32) ** 2))
        return {group: jnp.stack(per_slot) for group, per_slot in norms.items()}

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        out: dict[str, Array] = {}
        for shape, (Vs, Us) in vu.stacks.items():
            Ws = jnp.stack(
                [
                    _frozen_site_weight(self.target, name)
                    for name, s, _slot in vu.site_slots
                    if s == shape
                ]
            )
            out[shape] = Ws.astype(jnp.float32) - jnp.einsum(
                "gic,gco->goi", Vs.astype(jnp.float32), Us.astype(jnp.float32)
            )
        return out


def load_memory_toy_target(path: str, arch: str) -> MemoryToyTarget:
    """Load the frozen target from the trainer's safetensors (family-style key names)."""
    from safetensors import safe_open

    assert arch in ARCH_SITES, arch
    handle = safe_open(str(path), framework="numpy")
    get = lambda key: jnp.asarray(np.asarray(handle.get_tensor(key)), jnp.float32)
    is_attn, is_twoemb, is_mix = arch == "attn", arch == "twoemb", arch == "mix"
    target = MemoryToyTarget(
        arch=arch,
        wte=get("wte.weight"),
        lm_head=get("lm_head.weight"),
        g2=get("h.0.rms_2.weight"),
        gf=get("ln_f.weight"),
        q=get("h.0.attn.q_proj.weight") if is_attn else None,
        k=get("h.0.attn.k_proj.weight") if is_attn else None,
        v=get("h.0.attn.v_proj.weight") if is_attn else None,
        o=get("h.0.attn.o_proj.weight") if is_attn else None,
        g1=get("h.0.rms_1.weight") if is_attn else None,
        sinks=get("h.0.attn.sinks") if is_attn else None,
        wte0=get("wte0.weight") if is_twoemb else None,
        mix=get("mix.weight") if is_mix else None,
        fc=get("h.0.mlp.c_fc.weight"),
        down=get("h.0.mlp.down_proj.weight"),
    )
    assert target.wte.shape == (VOCAB, D) and target.fc.shape == (M, D), (
        target.wte.shape, target.fc.shape,
    )
    return target


def memory_toy_decomposed_model(
    arch: str, target: MemoryToyTarget, site_cs: tuple[SiteC, ...]
) -> MemoryToyDecomposedModel:
    return MemoryToyDecomposedModel(
        target=target, sites=site_specs(arch, site_cs), has_position_axis=False
    )


def replicate_target(target, mesh: Mesh):
    """Replicate the tiny frozen weights on every device (works for the target or the
    whole decomposed model)."""
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, target)


def neuron_aligned_initializer(
    model: MemoryToyDecomposedModel, key: PRNGKeyArray
) -> ComponentStacks:
    """Neuron/channel-aligned V/U init (the C recipe's `neuron_aligned`), built from the
    architecture-neutral factor helper: units on the output side for q/k/v/c_fc/mix,
    the input side for o_proj/down_proj."""
    vu: dict[str, tuple[Array, Array]] = {}
    for i, spec in enumerate(model.sites):
        W = _frozen_site_weight(model.target, spec.name)
        vu[spec.name] = _neuron_aligned_site_factors(
            W, spec, _UNITS_ON_INPUT.get(spec.name, False), jax.random.fold_in(key, i)
        )
    return component_stacks_from_site_arrays(model.sites, vu)
