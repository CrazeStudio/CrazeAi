import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from model import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE


# jax.tree_util.tree_flatten sorts dictionary keys alphabetically. Keep this
# layout in sync with the dictionaries built in training_jax/model.py.
# Source matrices are stored for x @ weight, while HF Linear layers expect
# [out_features, in_features], so every matrix must be transposed on export.
LAYER_TREE_LAYOUT = [
    ("down", "mlp.down_proj.weight", True),
    ("gate", "mlp.gate_proj.weight", True),
    ("k", "self_attn.k_proj.weight", True),
    ("ln1", "input_layernorm.weight", False),
    ("ln2", "post_attention_layernorm.weight", False),
    ("o", "self_attn.o_proj.weight", True),
    ("q", "self_attn.q_proj.weight", True),
    ("up", "mlp.up_proj.weight", True),
    ("v", "self_attn.v_proj.weight", True),
]


def as_fp16(arr, transpose=False):
    if transpose:
        arr = arr.T
    # safetensors requires C-contiguous NumPy arrays.
    return np.ascontiguousarray(arr.astype(np.float16))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(ckpt / "params.npz")

    expected_param_count = 1 + NUM_LAYERS * len(LAYER_TREE_LAYOUT) + 2
    expected_keys = {f"p{i}" for i in range(expected_param_count)}
    if set(z.files) != expected_keys:
        raise RuntimeError(
            f"Checkpoint should contain {expected_param_count} ordered tensors "
            f"(p0..p{expected_param_count - 1}); found {len(z.files)} tensors"
        )

    tensors = {}
    i = 0

    embedding = z[f"p{i}"]
    i += 1
    if embedding.shape != (VOCAB_SIZE, HIDDEN_SIZE):
        raise RuntimeError(
            f"Embedding shape mismatch: expected {(VOCAB_SIZE, HIDDEN_SIZE)}, "
            f"got {embedding.shape}"
        )
    tensors["model.embed_tokens.weight"] = as_fp16(embedding)

    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        for _source_key, hf_key, transpose in LAYER_TREE_LAYOUT:
            arr = z[f"p{i}"]
            i += 1
            tensors[f"{prefix}.{hf_key}"] = as_fp16(arr, transpose=transpose)

    # The flattened top-level dictionary order is embed, layers, lm_head, norm.
    # lm_head is [hidden, vocab] in JAX and must be transposed for Hugging Face.
    lm_head = z[f"p{i}"]
    i += 1
    norm = z[f"p{i}"]
    i += 1
    if lm_head.shape != (HIDDEN_SIZE, VOCAB_SIZE):
        raise RuntimeError(
            f"LM head shape mismatch: expected {(HIDDEN_SIZE, VOCAB_SIZE)}, "
            f"got {lm_head.shape}"
        )
    if norm.shape != (HIDDEN_SIZE,):
        raise RuntimeError(
            f"Final norm shape mismatch: expected {(HIDDEN_SIZE,)}, got {norm.shape}"
        )
    tensors["lm_head.weight"] = as_fp16(lm_head, transpose=True)
    tensors["model.norm.weight"] = as_fp16(norm)

    # Catch mismatches before invoking llama.cpp's converter.
    expected_shapes = {
        "model.embed_tokens.weight": (VOCAB_SIZE, HIDDEN_SIZE),
        "lm_head.weight": (VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": (HIDDEN_SIZE,),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        expected_shapes.update({
            f"{prefix}.self_attn.q_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
            f"{prefix}.self_attn.k_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
            f"{prefix}.self_attn.v_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
            f"{prefix}.self_attn.o_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
            f"{prefix}.mlp.gate_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            f"{prefix}.mlp.up_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            f"{prefix}.mlp.down_proj.weight": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            f"{prefix}.input_layernorm.weight": (HIDDEN_SIZE,),
            f"{prefix}.post_attention_layernorm.weight": (HIDDEN_SIZE,),
        })
    if set(tensors) != set(expected_shapes):
        raise RuntimeError("Exported tensor names do not match the expected Llama architecture")
    for name, shape in expected_shapes.items():
        if tensors[name].shape != shape:
            raise RuntimeError(
                f"{name} shape mismatch: expected {shape}, got {tensors[name].shape}"
            )

    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "float16",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = ckpt / name
        if src.exists():
            (out / name).write_bytes(src.read_bytes())
    print(f"Exported {len(tensors)} shape-validated tensors to {out}")


if __name__ == "__main__":
    main()
