import argparse
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import sentencepiece as spm
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file

from model_jax import CrazeTransformer


CONFIG = {
    "vocab_size": 16384,
    "dim": 896,
    "layers": 8,
    "heads": 14,
    "hidden": 3584,
    "max_length": 512,
}


def build_token_stream(data_path, tokenizer, output_path, max_examples=0):
    count = 0
    examples = 0
    with open(data_path, encoding="utf-8") as src, open(output_path, "wb") as dst:
        for line in src:
            if max_examples and examples >= max_examples:
                break
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            if not text.strip():
                continue
            ids = tokenizer.encode(text, out_type=int) + [3]
            np.asarray(ids, dtype=np.int32).tofile(dst)
            count += len(ids)
            examples += 1
    return count, examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-examples", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    token_file = out / "tokens.bin"
    token_count, example_count = build_token_stream(args.data, tokenizer, token_file, args.max_examples)

    raw = np.memmap(token_file, dtype=np.int32, mode="r")
    windows = max(1, (len(raw) - args.seq_len - 1) // args.seq_len)
    steps = args.steps or max(1, windows // args.batch_size)

    key = jax.random.PRNGKey(42)
    model = CrazeTransformer(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        layers=CONFIG["layers"],
        heads=CONFIG["heads"],
        hidden=CONFIG["hidden"],
    )
    dummy = jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]
    optimizer = optax.adamw(args.lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
    opt_state = optimizer.init(params)

    def loss_fn(p, x, y):
        logits = model.apply({"params": p}, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(p, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    print("JAX backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print("CrazeTransformer: random initialization; no pretrained base model")
    print("Examples:", example_count, "Tokens:", token_count, "Steps:", steps)

    for step_index in range(steps):
        xs, ys = [], []
        start = (step_index * args.batch_size * args.seq_len) % max(1, len(raw) - args.seq_len - 1)
        for b in range(args.batch_size):
            pos = (start + b * args.seq_len) % max(1, len(raw) - args.seq_len - 1)
            xs.append(np.asarray(raw[pos : pos + args.seq_len]))
            ys.append(np.asarray(raw[pos + 1 : pos + args.seq_len + 1]))
        params, opt_state, loss = step(params, opt_state, jnp.asarray(xs), jnp.asarray(ys))
        if step_index % 10 == 0:
            print(f"step {step_index + 1}/{steps} loss={float(loss):.4f}")

    flat = flatten_dict(params, sep=".")
    tensors = {}
    for name, value in flat.items():
        value = np.asarray(value)
        name = name.replace("layers_", "layers.")
        if name.endswith(".kernel"):
            name = name[:-7] + ".weight"
            value = value.T
        elif name.endswith(".scale"):
            name = name[:-6] + ".weight"
        elif name == "embed_tokens":
            name = "embed_tokens.weight"
        tensors["model." + name] = value

    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": CONFIG["vocab_size"],
        "hidden_size": CONFIG["dim"],
        "intermediate_size": CONFIG["hidden"],
        "num_hidden_layers": CONFIG["layers"],
        "num_attention_heads": CONFIG["heads"],
        "num_key_value_heads": CONFIG["heads"],
        "hidden_act": "silu",
        "max_position_embeddings": CONFIG["max_length"],
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")
    shutil.copy2(args.tokenizer, out / "tokenizer.model")
    (out / "training_info.json").write_text(
        json.dumps(
            {
                "architecture": "CrazeTransformer",
                "from_scratch": True,
                "pretrained_base": None,
                "backend": jax.default_backend(),
                "examples": example_count,
                "tokens": token_count,
                "steps": steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
