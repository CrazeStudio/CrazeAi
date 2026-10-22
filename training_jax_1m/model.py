import math
from typing import Any

import jax
import jax.numpy as jnp


VOCAB_SIZE = 2048
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 384
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
ROPE_THETA = 10000.0


def init_model(key: jax.Array, dtype=jnp.bfloat16):
    keys = jax.random.split(key, NUM_LAYERS * 8 + 2)
    i = 0

    def normal(k, shape, scale):
        return (jax.random.normal(k, shape, dtype=jnp.float32) * scale).astype(dtype)

    params = {
        "embed": normal(keys[i], (VOCAB_SIZE, HIDDEN_SIZE), 0.02)
    }
    i += 1
    layers = []
    attn_scale = 1.0 / math.sqrt(HIDDEN_SIZE)
    for _ in range(NUM_LAYERS):
        layer = {
            "q": normal(keys[i], (HIDDEN_SIZE, HIDDEN_SIZE), attn_scale),
            "k": normal(keys[i + 1], (HIDDEN_SIZE, HIDDEN_SIZE), attn_scale),
            "v": normal(keys[i + 2], (HIDDEN_SIZE, HIDDEN_SIZE), 0.02),
            "o": normal(keys[i + 3], (HIDDEN_SIZE, HIDDEN_SIZE), 0.02),
            "gate": normal(keys[i + 4], (HIDDEN_SIZE, INTERMEDIATE_SIZE), 0.02),
            "up": normal(keys[i + 5], (HIDDEN_SIZE, INTERMEDIATE_SIZE), 0.02),
            "down": normal(keys[i + 6], (INTERMEDIATE_SIZE, HIDDEN_SIZE), 0.02),
            "ln1": jnp.ones((HIDDEN_SIZE,), dtype=dtype),
            "ln2": jnp.ones((HIDDEN_SIZE,), dtype=dtype),
        }
        layers.append(layer)
        i += 7
    params["layers"] = layers
    params["norm"] = jnp.ones((HIDDEN_SIZE,), dtype=dtype)
    params["lm_head"] = normal(keys[i], (HIDDEN_SIZE, VOCAB_SIZE), 0.02)
    return params


def rms_norm(x, weight, eps=1e-5):
    # Keep the reduction in fp32 for stability, then return the model dtype.
    variance = jnp.mean(x.astype(jnp.float32) * x.astype(jnp.float32), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps) * weight).astype(x.dtype)


def rope(x, positions):
    # x: [B, S, H, D]
    half = x.shape[-1] // 2
    freq = 1.0 / (ROPE_THETA ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions[:, :, None, None] * freq[None, None, None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)


def causal_attention(x, layer, positions):
    b, s, _ = x.shape
    q = (x @ layer["q"]).reshape(b, s, NUM_HEADS, HEAD_DIM)
    k = (x @ layer["k"]).reshape(b, s, NUM_HEADS, HEAD_DIM)
    v = (x @ layer["v"]).reshape(b, s, NUM_HEADS, HEAD_DIM)

    q = rope(q, positions)
    k = rope(k, positions)

    # Let JAX/XLA select an optimized GPU attention implementation instead of
    # explicitly materializing the full [B,H,S,S] score/probability tensors.
    out = jax.nn.dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        scale=1.0 / math.sqrt(HEAD_DIM),
    )
    out = out.reshape(b, s, HIDDEN_SIZE)
    return out @ layer["o"]


def forward(params, tokens):
    x = params["embed"][tokens]
    positions = jnp.arange(tokens.shape[1], dtype=jnp.float32)[None, :]
    for layer in params["layers"]:
        h = rms_norm(x, layer["ln1"])
        x = x + causal_attention(h, layer, positions)
        h = rms_norm(x, layer["ln2"])
        h = jax.nn.silu(h @ layer["gate"]) * (h @ layer["up"])
        x = x + h @ layer["down"]
    x = rms_norm(x, params["norm"])
    return x @ params["lm_head"]


def loss_fn(params, tokens):
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    logits = forward(params, inputs)
    # Compute the loss in fp32 for numerical stability.
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked)
