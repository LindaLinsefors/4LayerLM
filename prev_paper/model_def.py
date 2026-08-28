"""Standalone LlamaSimpleMLP definition, runnable on Python 3.11.

Adapted from param-decomp-vpd/param_decomp/pretrain/models/llama_simple_mlp.py
(the code that trained the models), with the library dependencies removed and
training-only methods dropped. The forward pass is unchanged, including the
NewGELU (tanh-approximation) nonlinearity.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


@dataclass
class LlamaSimpleMLPConfig:
    model_type: str
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_intermediate: int
    mlp_bias: bool
    attn_bias: bool
    rotary_adjacent_pairs: bool
    rotary_dim: int
    rotary_base: int
    n_ctx: int
    n_key_value_heads: int
    use_grouped_query_attention: bool
    flash_attention: bool
    rms_norm_eps: float


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.use_grouped_query_attention, "only the GQA branch is ported"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_key_value_heads = config.n_key_value_heads
        self.repeat_kv_heads = config.n_head // config.n_key_value_heads
        self.rotary_dim = self.head_dim  # as in the original: rotary covers the full head
        self.rotary_adjacent_pairs = config.rotary_adjacent_pairs
        self.n_ctx = config.n_ctx
        self.flash_attention = config.flash_attention

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        self.k_proj = nn.Linear(
            config.n_embd, self.n_key_value_heads * self.head_dim, bias=config.attn_bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_key_value_heads * self.head_dim, bias=config.attn_bias
        )
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
        sin, cos = self._calculate_sin_cos_rotary(self.rotary_dim, self.n_ctx, config.rotary_base)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self.register_buffer("rotary_cos", cos, persistent=False)

    def _calculate_sin_cos_rotary(
        self, rotary_dim: int, n_ctx: int, base: int
    ) -> tuple[Tensor, Tensor]:
        pos = torch.arange(n_ctx, dtype=torch.float32)
        dim = torch.arange(rotary_dim // 2, dtype=torch.float32)
        freq = base ** (dim / (rotary_dim / 2))
        if self.rotary_adjacent_pairs:
            freq = freq.unsqueeze(1).repeat(1, 2).flatten()
        else:
            freq = freq.repeat(2)
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles), torch.cos(angles)

    def _rotate_every_two(self, x: Tensor) -> Tensor:
        x_rot = x.clone()
        if self.rotary_adjacent_pairs:
            x_rot[..., ::2] = -x[..., 1::2]
            x_rot[..., 1::2] = x[..., ::2]
        else:
            n = x.shape[-1] // 2
            x_rot[..., :n] = -x[..., n:]
            x_rot[..., n:] = x[..., :n]
        return x_rot

    def _apply_rotary_pos_emb(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor]:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, rotary_dim), broadcasts over heads
        sin = sin.unsqueeze(1)
        q_rotated = (q * cos) + (self._rotate_every_two(q) * sin)
        k_rotated = (k * cos) + (self._rotate_every_two(k) * sin)
        return q_rotated.to(q.dtype), k_rotated.to(k.dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.n_ctx - 1)
        cos = self.rotary_cos[position_ids].to(q.dtype)
        sin = self.rotary_sin[position_ids].to(q.dtype)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class NewGELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
            )
        )


class MLP(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_intermediate, bias=config.mlp_bias)
        self.gelu = NewGELU()
        self.down_proj = nn.Linear(config.n_intermediate, config.n_embd, bias=config.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gelu(self.c_fc(x)))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Block(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.rms_1 = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class LlamaSimpleMLP(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # tied embedding/unembedding

    def forward(self, idx: Tensor) -> Tensor:
        """Token ids (batch, pos) -> logits (batch, pos, vocab)."""
        _b, t = idx.size()
        assert t <= self.config.block_size
        x = self.wte(idx)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        is_1d = idx.dim() == 1
        if is_1d:
            idx = idx.unsqueeze(0)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits = self(idx_cond)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx.squeeze(0) if is_1d else idx
