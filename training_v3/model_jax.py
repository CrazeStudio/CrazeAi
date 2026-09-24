import math

import jax
import jax.numpy as jnp
from flax import linen as nn


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps) * scale


def rope(x, positions, theta=10000.0):
    half = x.shape[-1] // 2
    freq = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) / half))
    angles = positions[..., None] * freq
    c = jnp.cos(angles)[..., None, :]
    s = jnp.sin(angles)[..., None, :]
    a, b = x[..., :half], x[..., half:2 * half]
    return jnp.concatenate((a * c - b * s, a * s + b * c, x[..., 2 * half:]), axis=-1)


class Attention(nn.Module):
    dim: int
    heads: int

    @nn.compact
    def __call__(self, x, positions, mask):
        head_dim = self.dim // self.heads
        q = nn.Dense(self.dim, use_bias=False, name="q_proj")(x)
        k = nn.Dense(self.dim, use_bias=False, name="k_proj")(x)
        v = nn.Dense(self.dim, use_bias=False, name="v_proj")(x)
        q = rope(q.reshape(x.shape[:-1] + (self.heads, head_dim)), positions)
        k = rope(k.reshape(x.shape[:-1] + (self.heads, head_dim)), positions)
        v = v.reshape(x.shape[:-1] + (self.heads, head_dim))
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
        scores = jnp.where(mask[:, None, :, :], scores, jnp.finfo(scores.dtype).min)
        probs = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(x.shape)
        return nn.Dense(self.dim, use_bias=False, name="o_proj")(out)


class MLP(nn.Module):
    dim: int
    hidden: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.hidden, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(self.hidden, use_bias=False, name="up_proj")(x)
        return nn.Dense(self.dim, use_bias=False, name="down_proj")(jax.nn.silu(gate) * up)


class Block(nn.Module):
    dim: int
    heads: int
    hidden: int

    @nn.compact
    def __call__(self, x, positions, mask):
        h = RMSNorm(self.dim, name="input_layernorm")(x)
        x = x + Attention(self.dim, self.heads, name="self_attn")(h, positions, mask)
        h = RMSNorm(self.dim, name="post_attention_layernorm")(x)
        return x + MLP(self.dim, self.hidden, name="mlp")(h)


class CrazeTransformer(nn.Module):
    """CrazeAI decoder-only transformer, randomly initialized from scratch."""

    vocab_size: int = 16384
    dim: int = 896
    layers: int = 8
    heads: int = 14
    hidden: int = 3584

    @nn.compact
    def __call__(self, tokens):
        batch, length = tokens.shape
        embeddings = self.param(
            "embed_tokens", nn.initializers.normal(0.02), (self.vocab_size, self.dim)
        )
        x = embeddings[tokens]
        positions = jnp.arange(length, dtype=jnp.float32)[None, :]
        positions = jnp.broadcast_to(positions, (batch, length))
        mask = jnp.tril(jnp.ones((length, length), dtype=bool))
        mask = jnp.broadcast_to(mask, (batch, length, length))
        for i in range(self.layers):
            x = Block(self.dim, self.heads, self.hidden, name=f"layers_{i}")(x, positions, mask)
        x = RMSNorm(self.dim, name="norm")(x)
        return jnp.einsum("btd,vd->btv", x, embeddings)
