import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from model import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE, init_model, loss_fn

SPECIAL = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]
ROLE_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>")
FORMAT_VERSION = 2


def read_texts(path, limit=None):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if text:
                texts.append(text)
                if limit and len(texts) >= limit:
                    break
    return texts


def write_tokenizer_metadata(out_path):
    root = out_path.parent
    metadata = {
        "tokenizer_class": "GPT2TokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<|end|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": list(ROLE_TOKENS),
        "model_max_length": 2048,
    }
    (root / "tokenizer_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    special_map = {
        "bos_token": "<bos>",
        "eos_token": "<|end|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": list(ROLE_TOKENS),
    }
    (root / "special_tokens_map.json").write_text(json.dumps(special_map, indent=2), encoding="utf-8")


def build_tokenizer(data_path, out_path, train_limit=100000):
    out_path = Path(out_path)
    if out_path.exists():
        tok = Tokenizer.from_file(str(out_path))
        missing = [token for token in SPECIAL if tok.token_to_id(token) is None]
        if missing:
            raise RuntimeError(f"Cached tokenizer is missing required special tokens: {missing}")
        write_tokenizer_metadata(out_path)
        return tok

    texts = read_texts(data_path, train_limit)
    tok = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=1, special_tokens=SPECIAL)
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    write_tokenizer_metadata(out_path)
    actual_vocab = tok.get_vocab_size()
    if actual_vocab != VOCAB_SIZE:
        raise RuntimeError(f"Tokenizer produced {actual_vocab} tokens, expected {VOCAB_SIZE}")
    missing = [token for token in SPECIAL if tok.token_to_id(token) is None]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required special tokens: {missing}")
    return tok


def pack_dataset(data_path, tokenizer, packed_path, max_examples=None, seq_len=128):
    packed_path = Path(packed_path)
    mask_path = packed_path.with_name("packed_loss_mask.bin")
    meta_path = packed_path.with_suffix(".json")
    if packed_path.exists() and mask_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("format_version") == FORMAT_VERSION
            and meta.get("seq_len") == seq_len
            and meta.get("max_examples") == max_examples
        ):
            return (
                np.memmap(packed_path, dtype=np.uint16, mode="r", shape=(meta["tokens"],)),
                np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(meta["tokens"],)),
                meta,
            )

    eos_id = tokenizer.token_to_id("<eos>")
    end_id = tokenizer.token_to_id("<|end|>")
    role_ids = {token: tokenizer.token_to_id(token) for token in ROLE_TOKENS}
    if eos_id is None or end_id is None or any(v is None for v in role_ids.values()):
        raise RuntimeError("Tokenizer is missing required end-of-turn or role token IDs")
    assistant_id = role_ids["<|assistant|>"]
    role_id_values = set(role_ids.values())

    token_chunks = []
    mask_chunks = []
    seen = 0
    batch = []

    def flush_batch(texts):
        if not texts:
            return
        for encoded in tokenizer.encode_batch(texts):
            ids = encoded.ids
            active_assistant = False
            token_mask = []
            for token_id in ids:
                if token_id in role_id_values:
                    active_assistant = token_id == assistant_id
                    token_mask.append(0)
                elif token_id == end_id:
                    token_mask.append(int(active_assistant))
                    active_assistant = False
                else:
                    token_mask.append(int(active_assistant))
            token_chunks.append(np.asarray(ids + [eos_id], dtype=np.uint16))
            mask_chunks.append(np.asarray(token_mask + [0], dtype=np.uint8))

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not text:
                continue
            batch.append(text)
            seen += 1
            if len(batch) >= 256:
                flush_batch(batch)
                batch = []
            if max_examples and seen >= max_examples:
                break
    flush_batch(batch)

    if not token_chunks:
        raise RuntimeError("No training tokens were produced from the prepared dataset")
    stream = np.concatenate(token_chunks)
    mask_stream = np.concatenate(mask_chunks)
    usable = (len(stream) // (seq_len + 1)) * (seq_len + 1)
    stream = stream[:usable]
    mask_stream = mask_stream[:usable]
    if usable == 0 or not np.any(mask_stream):
        raise RuntimeError("No assistant-answer target tokens remain after packing and masking")

    packed_path.parent.mkdir(parents=True, exist_ok=True)
    token_mm = np.memmap(packed_path, dtype=np.uint16, mode="w+", shape=stream.shape)
    token_mm[:] = stream
    token_mm.flush()
    mask_mm = np.memmap(mask_path, dtype=np.uint8, mode="w+", shape=mask_stream.shape)
    mask_mm[:] = mask_stream
    mask_mm.flush()
    meta = {
        "format_version": FORMAT_VERSION,
        "tokens": int(len(stream)),
        "assistant_target_tokens": int(mask_stream.sum()),
        "seq_len": seq_len,
        "examples": int(seen),
        "max_examples": max_examples,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return (
        np.memmap(packed_path, dtype=np.uint16, mode="r", shape=(meta["tokens"],)),
        np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(meta["tokens"],)),
        meta,
    )


def save_params(params, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    np.savez(out_dir / "params.npz", **{f"p{i}": np.asarray(x) for i, x in enumerate(leaves)})
    (out_dir / "treedef.txt").write_text(str(treedef), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/sft.jsonl")
    p.add_argument("--output", default="artifacts/crazeai-10m-jax")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--chunk-size", type=int, default=8, help="Training steps compiled into one JAX dispatch")
    args = p.parse_args()

    print("JAX devices:", jax.devices())
    backend = jax.default_backend()
    default_batch = {"tpu": 32, "gpu": 8}.get(backend, 4)
    batch_size = args.batch_size or default_batch
    print(f"Backend={backend} batch_size={batch_size} seq_len={args.seq_len} chunk_size={args.chunk_size}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args.data, out / "tokenizer.json")
    packed, loss_mask, meta = pack_dataset(
        args.data, tokenizer, out / "packed_tokens.bin", args.max_examples, args.seq_len
    )
    nseq = meta["tokens"] // (args.seq_len + 1)
    if nseq < batch_size:
        raise RuntimeError("Dataset is too small for the selected batch size")

    param_dtype = jnp.bfloat16 if backend in ("gpu", "tpu") else jnp.float32
    params = init_model(jax.random.PRNGKey(42), dtype=param_dtype)
    param_count = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    if not 9_000_000 <= param_count <= 11_000_000:
        raise RuntimeError(f"Expected a roughly 10M parameter model, got {param_count:,}")

    tx = optax.adamw(args.lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
    opt_state = tx.init(params)

    def one_step(carry, batch_data):
        params, opt_state = carry
        tokens, answer_mask = batch_data
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, answer_mask)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def train_chunk(params, opt_state, tokens, masks):
        (params, opt_state), losses = jax.lax.scan(
            one_step, (params, opt_state), (tokens, masks)
        )
        return params, opt_state, losses

    @jax.jit
    def train_single(params, opt_state, tokens, masks):
        (params, opt_state), loss = one_step((params, opt_state), (tokens, masks))
        return params, opt_state, loss

    steps_per_epoch = nseq // batch_size
    print(
        f"Packed tokens={meta['tokens']:,} sequences={nseq:,} "
        f"assistant target tokens={meta['assistant_target_tokens']:,} "
        f"steps/epoch={steps_per_epoch:,}"
    )
    step = 0
    start = time.time()
    chunk = max(1, args.chunk_size)

    for epoch in range(args.epochs):
        order = np.arange(nseq)
        np.random.default_rng(42 + epoch).shuffle(order)
        b = 0
        while b < steps_per_epoch:
            remaining = steps_per_epoch - b
            take = min(chunk, remaining)
            idx = order[b * batch_size:(b + take) * batch_size]
            offsets = idx[:, None] * (args.seq_len + 1)
            positions = np.arange(args.seq_len + 1, dtype=np.int64)[None, :]
            batch = packed[offsets + positions].astype(np.int32).reshape(
                take, batch_size, args.seq_len + 1
            )
            masks = loss_mask[offsets + positions].astype(np.float32).reshape(
                take, batch_size, args.seq_len + 1
            )
            batch_jax = jnp.asarray(batch)
            masks_jax = jnp.asarray(masks)

            if take == chunk:
                params, opt_state, losses = train_chunk(params, opt_state, batch_jax, masks_jax)
                last_loss = float(losses[-1])
            else:
                for i in range(take):
                    params, opt_state, last_loss = train_single(
                        params, opt_state, batch_jax[i], masks_jax[i]
                    )
                    last_loss = float(last_loss)

            step += take
            b += take
            if step % 25 < take or step % 25 == 0:
                elapsed = time.time() - start
                print(f"epoch={epoch + 1}/{args.epochs} step={step} loss={last_loss:.4f} steps/s={step/elapsed:.3f}", flush=True)
            if step % 500 < take or step % 500 == 0:
                save_params(params, out)

    save_params(params, out)
    tokenizer.save(str(out / "tokenizer.json"))
    write_tokenizer_metadata(out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
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
    }, indent=2), encoding="utf-8")
    print("CrazeAI 10M response-masked training complete")


if __name__ == "__main__":
    main()
