# Adapted from "Abs in Sup"/circles.py for the 4LayerLM project.
#
# Probes the local geometry of the residual stream: how the map from
# post-embedding activations h0 to pre-unembedding activations behaves under
# perturbations (output change along random directions, Jacobian Frobenius norms).
#
# Differences from the original:
# * Models are the LlamaSimpleMLP transformers from load.py, not ResNets from
#   resnets.py. The analog of net.layers is layers_fn(net): the transformer
#   blocks + final RMSNorm applied to a single-token sequence (so attention
#   only attends to the token itself).
# * All analysis functions take in_layer/out_layer picking the residual-stream
#   points where the variation is applied and measured. Layers count in halves:
#   0 = just after embedding, 0.5 = after layer-1 attention (before its MLP),
#   1 = just after layer 1, ..., n_layer = after the last layer (+ final norm).
# * The analog of net.emb(one_hot) is the embedding row net.wte.weight[token_id].
# * Jacobians are computed in chunks: at res_dim=768 a full grid of per-sample
#   (768, 768) Jacobians would not fit in memory.
#
# %%

import math

import matplotlib.pyplot as plt
import torch

import load


def _half_steps(net, layer):
    """Number of half-layer steps (attention or MLP sublayers) from the embedding
    to residual-stream point `layer`. Layers are counted in halves:
    0 = just after embedding, 0.5 = after attention of the first layer (before its
    MLP), 1 = after the first full layer, ... , n_layer = after the last layer.
    """
    steps = round(2 * layer)
    assert steps == 2 * layer, f"layer must be a multiple of 0.5, got {layer}"
    assert 0 <= steps <= 2 * net.config.n_layer, f"layer {layer} outside [0, {net.config.n_layer}]"
    return steps


def layers_fn(net, in_layer=0, out_layer=None, final_norm=None, sequence=False):
    """The residual-stream map of the transformer from the point `in_layer` to
    the point `out_layer` (see _half_steps for the layer coordinate convention).

    Defaults: in_layer=0 (just after embedding), out_layer=n_layer, and the
    final RMSNorm is applied only when out_layer is the very end of the stack
    (so the default reproduces "embedding -> just before unembedding").
    Pass final_norm=True/False to override.

    If sequence=False, each res_dim vector is treated as its own length-1
    sequence, so attention attends only to the token itself; the returned
    function maps (*batch_dims, res_dim) -> (*batch_dims, res_dim).
    If sequence=True, the last two dims are (seq_len, res_dim) and attention
    acts across positions (causally), exactly as in a normal forward pass; the
    returned function maps (*batch_dims, seq_len, res_dim) -> same shape.
    Any number of batch dims (including none) in both cases.

    net: a LlamaSimpleMLP
    returns: a function as described above
    """
    if out_layer is None:
        out_layer = net.config.n_layer
    assert in_layer <= out_layer, (in_layer, out_layer)
    first, last = _half_steps(net, in_layer), _half_steps(net, out_layer)
    if final_norm is None:
        final_norm = last == 2 * net.config.n_layer

    def f(h0):
        if not sequence:
            h0 = h0.unsqueeze(-2)  # each vector = a length-1 sequence
        batch_dims = h0.shape[:-2]
        x = h0.reshape(-1, *h0.shape[-2:])  # (flat_batch, pos, res_dim)
        for step in range(first, last):
            block = net.h[step // 2]
            if step % 2 == 0:
                x = x + block.attn(block.rms_1(x))
            else:
                x = x + block.mlp(block.rms_2(x))
        if final_norm:
            x = net.ln_f(x)
        x = x.reshape(*batch_dims, *x.shape[-2:])
        return x if sequence else x.squeeze(-2)

    return f


def text_h0(net, tokens, in_layer=0):
    """Residual-stream activations at in_layer for a tokenized text.

    net: a LlamaSimpleMLP
    tokens: token ids — a list[int] or 1-D int tensor, e.g. tokenizer.encode(text)
        or a row from load.pile_samples / load.simple_samples
    returns: tensor of shape (seq_len, res_dim) — one activation per token
        position, computed with a normal forward pass up to in_layer
    """
    ids = torch.as_tensor(tokens)
    h0 = net.wte(ids)
    with torch.no_grad():
        return layers_fn(net, 0, in_layer, final_norm=False, sequence=True)(h0).detach()


def res_dim(net):
    return net.config.n_embd


def token_h0(net, token_id):
    """Activation just after the embedding for a token id.

    token_id: int, or an integer tensor of shape (*batch_dims,)
    returns: tensor of shape (*batch_dims, res_dim)
    """
    return net.wte.weight[token_id].detach()


def _apply_rope(attn, x, positions):
    """Rotary embedding for x of shape (..., head_dim), at the given position(s).
    positions: an int, or an int tensor broadcastable against x's position dim."""
    cos = attn.rotary_cos[positions]
    sin = attn.rotary_sin[positions]
    return x * cos + attn._rotate_every_two(x) * sin


def _last_token_jacobian_norms(net, flat, in_layer, out_layer, chunk_size, final_norm=None):
    """Frobenius norm of d(residual at out_layer, last position) /
    d(residual at in_layer, last position) for each sample in
    flat (batch, seq_len, res_dim), with the prefix positions held fixed.

    Exploits causality: the prefix positions' activations do not depend on the
    last position, so each attention sublayer's prefix keys/values are computed
    once per sample without autodiff and treated as constants. Differentiation
    then runs on a single position, making its cost independent of seq_len.
    """
    if out_layer is None:
        out_layer = net.config.n_layer
    assert in_layer <= out_layer, (in_layer, out_layer)
    first, last = _half_steps(net, in_layer), _half_steps(net, out_layer)
    if final_norm is None:
        final_norm = last == 2 * net.config.n_layer
    pos = flat.shape[1] - 1

    def prefix_kv(prefix):  # (chunk, pos, res_dim) -> [K1, V1, K2, V2, ...]
        kvs = []
        x = prefix
        for step in range(first, last):
            block = net.h[step // 2]
            if step % 2 == 0:
                attn = block.attn
                xn = block.rms_1(x)
                c, p, _ = xn.shape
                k = attn.k_proj(xn).view(c, p, attn.n_key_value_heads, attn.head_dim).transpose(1, 2)
                v = attn.v_proj(xn).view(c, p, attn.n_key_value_heads, attn.head_dim).transpose(1, 2)
                kvs += [_apply_rope(attn, k, torch.arange(p)), v]
                x = x + attn(xn) if p > 0 else x
            else:
                x = x + block.mlp(block.rms_2(x)) if x.shape[1] > 0 else x
        return kvs

    def f_last(last_vec, *kvs):  # (res_dim,) + per-sample KV constants -> (res_dim,)
        x = last_vec
        kv_iter = iter(kvs)
        for step in range(first, last):
            block = net.h[step // 2]
            if step % 2 == 0:
                attn = block.attn
                xn = block.rms_1(x)
                q = _apply_rope(attn, attn.q_proj(xn).view(attn.n_head, attn.head_dim), pos)
                k = _apply_rope(attn, attn.k_proj(xn).view(attn.n_key_value_heads, attn.head_dim), pos)
                v = attn.v_proj(xn).view(attn.n_key_value_heads, attn.head_dim)
                K = torch.cat([next(kv_iter), k[:, None, :]], dim=1)  # (n_kv, pos+1, head_dim)
                V = torch.cat([next(kv_iter), v[:, None, :]], dim=1)
                if attn.repeat_kv_heads > 1:
                    K = K.repeat_interleave(attn.repeat_kv_heads, dim=0)
                    V = V.repeat_interleave(attn.repeat_kv_heads, dim=0)
                att = torch.softmax(torch.einsum('hd,hpd->hp', q, K) / math.sqrt(attn.head_dim), dim=-1)
                x = x + attn.o_proj(torch.einsum('hp,hpd->hd', att, V).reshape(-1))
            else:
                x = x + block.mlp(block.rms_2(x))
        if final_norm:
            x = net.ln_f(x)
        return x

    jac_fn = torch.vmap(torch.func.jacfwd(f_last, argnums=0))
    norms = []
    for chunk in flat.split(chunk_size):
        with torch.no_grad():
            kvs = prefix_kv(chunk[:, :-1])
        jac = jac_fn(chunk[:, -1].contiguous(), *kvs)  # jacfwd rejects overlapping-memory views
        norms.append(jac.flatten(start_dim=1).norm(dim=1))
    return torch.cat(norms)


def residual_jacobian_frobenius_norms(
    net, h0, in_layer=0, out_layer=None, chunk_size=256, sequence=True, vary='last', final_norm=None
):
    """For each sample in the batch, compute the Frobenius norm of the Jacobian
    of the residual stream activations at out_layer with respect to the
    activations at in_layer (defaults: embedding -> just before unembedding).

    net: a LlamaSimpleMLP
    h0: batch of activations at in_layer; any number of batch dims (including
        none). Each sample is a single vector (res_dim,) if sequence=False, or a
        whole sequence (seq_len, res_dim) — e.g. from text_h0 — if sequence=True.
    vary (only used if sequence=True): differentiate with respect to the whole
        sequence ('all') or only the last token position ('last'); the output
        side is always the last token position at out_layer.
    returns: tensor of shape (*batch_dims,) with one Frobenius norm per sample
    """
    def frob(jacobians):  # (chunk, *jac_dims) -> (chunk,), freeing the Jacobians
        return jacobians.flatten(start_dim=1).norm(dim=1)

    if not sequence:
        batch_dims = h0.shape[:-1]
        flat = h0.reshape(-1, res_dim(net))
        jac_fn = torch.vmap(torch.func.jacrev(layers_fn(net, in_layer, out_layer, final_norm=final_norm)))
        norms = [frob(jac_fn(chunk)) for chunk in flat.split(chunk_size)]
    else:
        batch_dims = h0.shape[:-2]
        flat = h0.reshape(-1, *h0.shape[-2:])
        seq_len = flat.shape[1]
        f = layers_fn(net, in_layer, out_layer, final_norm=final_norm, sequence=True)
        if vary == 'all':

            def f_last_out(h):
                return f(h)[-1]

            # each per-sample Jacobian is (d, seq_len, d): shrink chunks
            chunk_size = max(1, chunk_size // seq_len)
            jac_fn = torch.vmap(torch.func.jacrev(f_last_out))
            norms = [frob(jac_fn(chunk)) for chunk in flat.split(chunk_size)]
        elif vary == 'last':
            norms = [_last_token_jacobian_norms(net, flat, in_layer, out_layer, chunk_size, final_norm)]
        else:
            raise ValueError(f"vary must be 'all' or 'last', got {vary!r}")

    return torch.cat(norms).reshape(batch_dims)


def _random_unit_directions(n, sample_shape, vary):
    """n random unit directions in the perturbation space, shape (n, *sample_shape).
    sample_shape is (res_dim,) for single vectors or (seq_len, res_dim) for
    sequences; unit norm is over the whole sample. For sequences, vary='all'
    supports the direction on all positions, vary='last' only on the last one."""
    if len(sample_shape) == 1 or vary == 'all':
        directions = torch.randn(n, *sample_shape)
    elif vary == 'last':
        directions = torch.zeros(n, *sample_shape)
        directions[:, -1] = torch.randn(n, sample_shape[-1])
    else:
        raise ValueError(f"vary must be 'all' or 'last', got {vary!r}")
    norms = directions.flatten(start_dim=1).norm(dim=-1)
    return directions / norms.reshape(n, *[1] * len(sample_shape))


def plot_output_change_along_random_directions(
    net, h0, n, max_dist=1.0, n_points=100, in_layer=0, out_layer=None, sequence=True, vary='last',
    title=None, final_norm=None
):
    """Plot how much the residual stream at out_layer moves as h0 (the residual
    stream at in_layer) is perturbed along n random directions.

    For each of n random unit directions d, plots
    ||layers(h0 + t*d) - layers(h0)|| as a function of t in [0, max_dist],
    where layers = the map from in_layer to out_layer.
    All lines in a single plot, single color, no legend.

    net: a LlamaSimpleMLP
    h0: a single sample of activations at in_layer: a vector (res_dim,) if
        sequence=False, or a whole sequence (seq_len, res_dim) — e.g. from
        text_h0 — if sequence=True
    n: number of random directions
    vary (only used if sequence=True): perturb the whole sequence ('all') or
        only the last token position ('last'); the output change is always
        measured at the last token position at out_layer.
    """
    sample_ndim = 2 if sequence else 1
    assert h0.ndim == sample_ndim, f"expected a single sample, got shape {tuple(h0.shape)}"
    directions = _random_unit_directions(n, h0.shape, vary)

    t = torch.linspace(0, max_dist, n_points)

    f = layers_fn(net, in_layer, out_layer, sequence=sequence, final_norm=final_norm)
    t_bcast = t.reshape(n_points, *[1] * (1 + sample_ndim))
    with torch.no_grad():
        base = f(h0)
        # (n_points, n, *h0.shape): h0 perturbed by t*d for every (t, direction) pair
        perturbed = f(h0 + t_bcast * directions)
        diff = perturbed - base
        if sequence:
            diff = diff[..., -1, :]  # measure the change at the last position only
        dists = diff.norm(dim=-1)  # (n_points, n)

    plt.plot(t, dists, color='C0', alpha=min(1.0, 10 / n))
    plt.xlabel('distance from h0')
    plt.ylabel('||layers(h0 + t·d) - layers(h0)||')
    if title is not None:
        plt.title(title)
    plt.show()


def make_grid(h0, max_dist, n_points_per_dim=100, sequence=True, vary='last'):
    """Make a grid of points in a random plane in residual space around h0.

    h0: a single sample of activations: a vector (res_dim,) if sequence=False,
        or a whole sequence (seq_len, res_dim) if sequence=True — then the plane
        lives in the full sequence space ('all') or in the last token position
        only ('last')
    max_dist: maximum distance from h0 to include in the grid
    returns: tensor of shape (n_points_per_dim, n_points_per_dim, *h0.shape)
        with points in a random plane around h0
    """
    assert h0.ndim == (2 if sequence else 1), f"expected a single sample, got shape {tuple(h0.shape)}"
    # random plane spanned by two orthonormal directions
    d1, d2 = _random_unit_directions(2, h0.shape, vary)
    d2 = d2 - (d2 * d1).sum() * d1  # make d2 orthogonal to d1
    d2 = d2 / d2.norm()

    linspace = torch.linspace(-max_dist, max_dist, n_points_per_dim)
    grid = torch.stack(torch.meshgrid(linspace, linspace, indexing='ij'), dim=-1)
    expand = (...,) + (None,) * h0.ndim
    offsets = grid[..., 0][expand] * d1 + grid[..., 1][expand] * d2  # (n, n, *h0.shape)
    return h0 + offsets


def plot_residual_jacobian_frobenius_norms(
    net, h0, max_dist, n_points_per_dim=25, in_layer=0, out_layer=None, sequence=True, vary='last',
    title=None, final_norm=None
):
    """Plot the Frobenius norm of the Jacobian of the residual stream activations
    at out_layer with respect to the activations at in_layer (defaults:
    embedding -> just before unembedding), for a grid of points in a random
    plane around h0.

    net: a LlamaSimpleMLP
    h0: a single sample of activations at in_layer: a vector (res_dim,) if
        sequence=False, or a whole sequence (seq_len, res_dim) — e.g. from
        text_h0 — if sequence=True
    max_dist: maximum distance from h0 to include in the grid
    vary (only used if sequence=True): the grid plane and the differentiation
        are in the whole sequence space ('all') or the last token position only
        ('last').
    """
    grid = make_grid(h0, max_dist, n_points_per_dim, sequence, vary)
    norms = residual_jacobian_frobenius_norms(
        net, grid, in_layer, out_layer, sequence=sequence, vary=vary, final_norm=final_norm
    )
    # norms[i, j] has plane coordinates (u, v) = (linspace[i], linspace[j]):
    # transpose + origin='lower' puts u on the x-axis, v on the y-axis
    plt.imshow(
        norms.detach().numpy().T,
        origin='lower',
        extent=(-max_dist, max_dist, -max_dist, max_dist),
    )
    plt.xlabel('distance from h0 along d1')
    plt.ylabel('distance from h0 along d2')
    plt.colorbar(label='Frobenius norm of Jacobian of residual stream')
    if title is not None:
        plt.title(title)
    plt.show()


def text_change(
    net, tokens, n=100, max_dist=1.0, n_points=100, in_layer=0, out_layer=None, vary='last',
    title=None, final_norm=None,
):
    """plot_output_change_along_random_directions for a tokenized text: computes
    the residual stream at in_layer with text_h0 and perturbs it there.

    tokens: token ids — a list[int] or 1-D int tensor (see text_h0).
    See plot_output_change_along_random_directions for the other arguments.
    """
    h0 = text_h0(net, tokens, in_layer)
    plot_output_change_along_random_directions(
        net, h0, n, max_dist, n_points, in_layer, out_layer, sequence=True, vary=vary,
        title=title, final_norm=final_norm,
    )


def text_jacobian(
    net, tokens, max_dist=1.0, n_points_per_dim=25, in_layer=0, out_layer=None, vary='last',
    title=None, final_norm=None
):
    """plot_residual_jacobian_frobenius_norms for a tokenized text: computes the
    residual stream at in_layer with text_h0 and makes the grid around it there.

    tokens: token ids — a list[int] or 1-D int tensor (see text_h0).
    See plot_residual_jacobian_frobenius_norms for the other arguments.
    """
    h0 = text_h0(net, tokens, in_layer)
    plot_residual_jacobian_frobenius_norms(
        net, h0, max_dist, n_points_per_dim, in_layer, out_layer, sequence=True, vary=vary,
        title=title, final_norm=final_norm,
    )


# %%
'''

# The 2-layer SimpleStories model (res_dim=192) keeps the Jacobian grids cheap;
# swap in load.load_pile_4l() for the 4-layer Pile model (res_dim=768, ~16x slower).
net, _, _ = load.load_pile_4l()

token_0_h0 = token_h0(net, 0)

plot_output_change_along_random_directions(net, h0=token_0_h0, n=100, max_dist=0.1, n_points=100, sequence=False)
plot_output_change_along_random_directions(net, h0=token_0_h0, n=100, max_dist=1.0, n_points=100, sequence=False)
plot_output_change_along_random_directions(net, h0=token_0_h0, n=100, max_dist=10.0, n_points=100, sequence=False)

for max_dist in [0.1, 1.0, 10.0]:
    plot_residual_jacobian_frobenius_norms(net, h0=token_0_h0, max_dist=max_dist, sequence=False)

# %%
# Superposition of two token embeddings (the analog of the original's two-hot input)
two_token_h0 = token_h0(net, 0) + token_h0(net, 1)

plot_output_change_along_random_directions(net, h0=two_token_h0, n=100, max_dist=0.1, n_points=100, sequence=False)
plot_output_change_along_random_directions(net, h0=two_token_h0, n=100, max_dist=1.0, n_points=100, sequence=False)
plot_output_change_along_random_directions(net, h0=two_token_h0, n=100, max_dist=10.0, n_points=100, sequence=False)

for max_dist in [0.1, 1.0, 10.0]:
    plot_residual_jacobian_frobenius_norms(net, h0=two_token_h0, max_dist=max_dist, sequence=False)

# %%

print(two_token_h0.norm())
print(token_0_h0.norm())

rand_h0 = torch.randn(res_dim(net))
rand_h0 = rand_h0 / rand_h0.norm() * token_0_h0.norm()
print(rand_h0.norm())

# %%
plot_output_change_along_random_directions(net, h0=rand_h0, n=100, max_dist=0.1, n_points=100, sequence=False)
plot_output_change_along_random_directions(net, h0=rand_h0, n=100, max_dist=1.0, n_points=100, sequence=False)
plot_output_change_along_random_directions(net, h0=rand_h0, n=100, max_dist=10.0, n_points=100, sequence=False)

for max_dist in [0.1, 1.0, 10.0]:
    plot_residual_jacobian_frobenius_norms(net, h0=rand_h0, max_dist=max_dist, sequence=False)

# %%

h0 = rand_h0
max_dist = 5
plot_output_change_along_random_directions(net, h0=h0, n=100, max_dist=max_dist, n_points=100, sequence=False)
plot_residual_jacobian_frobenius_norms(net, h0=h0, max_dist=max_dist, n_points_per_dim=25, sequence=False)

# %%
'''