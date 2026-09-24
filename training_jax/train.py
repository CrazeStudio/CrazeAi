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

from model import VOCAB_SIZE, init_model, loss_fn

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


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
    (root / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 2048,
    }, indent=2))
    (root / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }, indent=2))


def build_tokenizer(data_path, out_path, train_limit=100000):
    out_path = Path(out_path)
    if out_path.exists():
        return Tokenizer.from_file(str(out_path))
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
    return tok


def pack_dataset(data_path, tokenizer, packed_path, max_examples=None, seq_len=512):
    packed_path = Path(packed_path)
    meta_path = packed_path.with_suffix(".json")
    if packed_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("seq_len") == seq_len and meta.get("max_examples") == max_examples:
            return np.memmap(packed_path, dtype=np.uint16, mode="r"), meta

    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise RuntimeError("Tokenizer has no <eos> token")

    chunks = []
    seen = 0
    with open(data_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not text:
                continue
            batch.append(text)
            seen += 1
            if len(batch) >= 2048 or (max_examples and seen >= max_examples):
                enc = tokenizer.encode_batch(batch)
                ids = []
                for e in enc:
                    ids.extend(e.ids)
                    ids.append(eos)
                chunks.append(np.asarray(ids, dtype=np.uint16))
                batch = []
            if max_examples and seen >= max_examples:
                break
        if batch:
            enc = tokenizer.encode_batch(batch)
            ids = []
            for e in enc:
                ids.extend(e.ids)
                ids.append(eos)
            chunks.append(np.asarray(ids, dtype=np.uint16))

    stream = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.uint16)
    usable = (len(stream) // (seq_len + 1)) * (seq_len + 1)
    stream = stream[:usable]
    packed_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(packed_path, dtype=np.uint16, mode="w+", shape=stream.shape)
    mm[:] = stream
    mm.flush()
    meta = {"tokens": int(len(stream)), "seq_len": seq_len, "examples": int(seen), "max_examples": max_examples}
    meta_path.write_text(json.dumps(meta, indent=2))
    return np.memmap(packed_path, dtype=np.uint16, mode="r"), meta


def save_params(params, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    np.savez(out_dir / "params.npz", **{f"p{i}": np.asarray(x) for i, x in enumerate(leaves)})
    (out_dir / "treedef.txt").write_text(str(treedef), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/sft.jsonl")
    p.add_argument("--output", default="artifacts/crazeai-jax")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    print("JAX devices:", jax.devices())
    backend = jax.default_backend()
    batch_size = args.batch_size or ({"tpu": 32, "gpu": 16}.get(backend, 4))
    print(f"Backend={backend} batch_size={batch_size} seq_len={args.seq_len}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"
    tokenizer = build_tokenizer(args.data, tok_path)
    data_path = out / "packed_tokens.bin"
    packed, meta = pack_dataset(args.data, tokenizer, data_path, args.max_examples, args.seq_len)
    nseq = meta["tokens"] // (args.seq_len + 1)
    if nseq < batch_size:
        raise RuntimeError("Dataset is too small for the selected batch size")

    params = init_model(jax.random.PRNGKey(42))
    tx = optax.adamw(args.lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    steps_per_epoch = nseq // batch_size
    print(f"Packed tokens={meta['tokens']:,} sequences={nseq:,} steps/epoch={steps_per_epoch:,}")
    step = 0
    start = time.time()
    for epoch in range(args.epochs):
        order = np.arange(nseq)
        np.random.default_rng(42 + epoch).shuffle(order)
        for b in range(steps_per_epoch):
            idx = order[b * batch_size:(b + 1) * batch_size]
            offsets = idx * (args.seq_len + 1)
            batch = np.stack([packed[o:o + args.seq_len + 1] for o in offsets]).astype(np.int32)
            params, opt_state, loss = train_step(params, opt_state, jnp.asarray(batch))
            step += 1
            if step % 25 == 0:
                elapsed = time.time() - start
                print(f"step={step} loss={float(loss):.4f} steps/s={step/elapsed:.3f}", flush=True)
            if step % 500 == 0:
                save_params(params, out)

    save_params(params, out)
    tokenizer.save(str(tok_path))
    write_tokenizer_metadata(tok_path)
    (out / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }, indent=2))
    print("CrazeAI JAX training complete")


if __name__ == "__main__":
    main()
