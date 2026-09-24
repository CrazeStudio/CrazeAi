import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from model import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS, VOCAB_SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(ckpt / "params.npz")

    tensors = {}
    i = 0
    tensors["model.embed_tokens.weight"] = z[f"p{i}"].astype(np.float16); i += 1
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "input_layernorm", "post_attention_layernorm"]
        for name in names:
            arr = z[f"p{i}"]
            i += 1
            if name in {"input_layernorm", "post_attention_layernorm"}:
                tensors[f"{prefix}.{name}.weight"] = arr.astype(np.float16)
            else:
                tensors[f"{prefix}.self_attn.{name}.weight" if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else f"{prefix}.mlp.{name}.weight"] = arr.T.astype(np.float16)
    tensors["model.norm.weight"] = z[f"p{i}"].astype(np.float16); i += 1
    tensors["lm_head.weight"] = z[f"p{i}"].astype(np.float16)

    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "float16",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = ckpt / name
        if src.exists():
            (out / name).write_bytes(src.read_bytes())
    print(f"Exported {len(tensors)} tensors to {out}")


if __name__ == "__main__":
    main()
