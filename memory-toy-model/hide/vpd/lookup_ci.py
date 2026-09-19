"""A trained LOOKUP-TABLE causal-importance function for the memory toy.

Diagnostic for the CI-capacity concern (2026-09-19): the global-MLP CI net (~9-11M
params ~ 21 Mbit) cannot represent an arbitrary per-fact sparse CI pattern
(n x C_total ~ 75-300 Mbit), so "everything always on" may be a CI bottleneck artifact.
Here CI is a trainable table: preactivation[site][fact, component], one row per training
fact — maximal CI expressivity, possible because train = eval = a finite fact table.

Plumbing trick: both classes SUBCLASS the GlobalMLP CI classes, so every closed
`case GlobalMLPCIFn()` / `case GlobalMLPCIArch()` match in core (placement, padding,
optimizer build, compute cast) accepts them via isinstance — only the construction seam
(`init_placed.build_ci_fn`) needs the wrapper installed by `install()`. The parent MLP
is kept as a tiny vestige (hidden [8], input = the fact-index tap) whose output is
unused; the tables are the trained CI parameters. The fact index reaches the CI fn as
a capture tap `fact_idx` recorded by the memory_toy forward (tokens column 2).
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn import (
    CI,
    CIFnPlacement,
    GlobalMLPCIArch,
    GlobalMLPCIFn,
    TapSpec,
    init_global_mlp_ci_fn,
)
from param_decomp.core.components import SiteSpec

FACT_IDX_KEY = "fact_idx"


@dataclass(frozen=True)
class LookupCIArch(GlobalMLPCIArch):
    n_facts: int = 0


class LookupTableCIFn(GlobalMLPCIFn):
    tables: tuple[Array, ...]  # one (n_facts, C) preactivation table per output site

    def shardings(self, mesh: Mesh) -> "LookupTableCIFn":
        base = super().shardings(mesh)
        repl = NamedSharding(mesh, P())
        return eqx.tree_at(lambda f: f.tables, base, tuple(repl for _ in self.tables))

    def __call__(
        self, taps: dict[str, Array], *, remat: bool, placement: CIFnPlacement | None
    ) -> CI:
        del remat
        assert placement is None, f"{type(self).__name__} is unplaced"
        digits = taps[FACT_IDX_KEY]  # (B, 2) base-256 digits, bf16-exact (see memory_toy)
        idx = (
            jnp.round(digits[..., 0]).astype(jnp.int32) * 256
            + jnp.round(digits[..., 1]).astype(jnp.int32)
        )
        match jax.typeof(idx).sharding:
            case NamedSharding(mesh=mesh, spec=spec):
                out = NamedSharding(mesh, P(spec[0] if len(spec) else None, None))
            case other:
                raise AssertionError(f"expected named sharding on fact_idx, got {type(other)}")
        pre = {
            name: table.at[idx].get(out_sharding=out)
            for name, table in zip(self.output_names, self.tables, strict=True)
        }
        return CI.from_preactivations(pre)


def build_lookup_ci_fn(
    arch: LookupCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> LookupTableCIFn:
    assert arch.n_facts > 0, arch
    parent = init_global_mlp_ci_fn(
        GlobalMLPCIArch(
            hidden_dims=arch.hidden_dims,
            has_position_axis=arch.has_position_axis,
            input_taps=arch.input_taps,
        ),
        sites,
        key,
    )
    # init at preactivation 1.0: every component fully on, matching how a fresh MLP CI
    # run effectively starts; the minimality losses prune from there
    tables = tuple(jnp.ones((arch.n_facts, s.C), jnp.float32) for s in sites)
    return LookupTableCIFn(
        mlp=parent.mlp,
        input_taps=parent.input_taps,
        output_names=parent.output_names,
        c_sizes=parent.c_sizes,
        has_position_axis=parent.has_position_axis,
        tables=tables,
    )


def lookup_ci_arch(n_facts: int, hidden_dims: tuple[int, ...] = (8,)) -> LookupCIArch:
    return LookupCIArch(
        hidden_dims=hidden_dims,
        has_position_axis=False,
        input_taps=(TapSpec(key=FACT_IDX_KEY, width=2),),
        n_facts=n_facts,
    )


def install() -> None:
    """Route `init_placed.build_ci_fn` through the lookup constructor for LookupCIArch
    (the one construction seam; every other core dispatch accepts the subclasses)."""
    import param_decomp.core.init_placed as init_placed

    if getattr(init_placed.build_ci_fn, "__lookup_wrapped__", False):
        return
    orig = init_placed.build_ci_fn

    def wrapper(arch, sites, key):
        if isinstance(arch, LookupCIArch):
            return build_lookup_ci_fn(arch, sites, key)
        return orig(arch, sites, key)

    wrapper.__lookup_wrapped__ = True
    init_placed.build_ci_fn = wrapper
