import math
from typing import Any

import jax
import jax.numpy as jnp


VOCAB_SIZE = 32768
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 2048
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
ROPE_THETA = 10000.0


def init_model(key: jax.Array):
    keys = jax.random.split(key, NUM_LAYERS * 8 + 2)
    i = 0

    def normal(k, shape, scale):
        return jax.random.normal(k, shape, dtype=jnp.float32) * scale

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
            "ln1": jnp.ones((HIDDEN_SIZE,), dtype=jnp.float32),
            "ln2": jnp.ones((HIDDEN_SIZE,), dtype=jnp.float32),
        }
        layers.append(layer)
        i += 7
    params["layers"] = layers
    params["norm"] = jnp.ones((HIDDEN_SIZE,), dtype=jnp.float32)
    params["lm_head"] = normal(keys[i], (HIDDEN_SIZE, VOCAB_SIZE), 0.02)
    return params


def rms_norm(x, weight, eps=1e-5):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope(x, positions):
    # x: [B, S, H, D]
    half = x.shape[-1] // 2
    freq = 1.0 / (ROPE_THETA ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions[:, :, None, None] * freq[None, None, None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def causal_attention(x, layer, positions):
    b, s, _ = x.shape
    q = x @ layer["q"]
    k = x @ layer["k"]
    v = x @ layer["v"]
    q = q.reshape(b, s, NUM_HEADS, HEAD_DIM)
    k = k.reshape(b, s, NUM_HEADS, HEAD_DIM)
    v = v.reshape(b, s, NUM_HEADS, HEAD_DIM)
    q = rope(q, positions)
    k = rope(k, positions)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(HEAD_DIM)
    mask = jnp.tril(jnp.ones((s, s), dtype=bool))[None, None, :, :]
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, s, HIDDEN_SIZE)
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
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return -jnp.mean(picked)
