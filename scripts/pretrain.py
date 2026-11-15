from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from crazeai.model import CrazeAIConfig, CrazeAIModel


class TokenBlocks(Dataset):
    def __init__(self, path: Path, block_size: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.count = max(0, (len(self.tokens) - 1) // block_size)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = np.asarray(
            self.tokens[start:start + self.block_size + 1], dtype=np.int64
        )
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def encode_corpus(
    corpus: Path, tokenizer_path: Path, output: Path,
    validation_fraction: float, seed: int
):
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()
    cache_manifest = output / "encoded_manifest.json"
    train_path, val_path = output / "train.bin", output / "validation.bin"
    corpus_stat = corpus.stat()
    expected = {
        "tokenizer_sha256": tokenizer_hash,
        "corpus_size": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "validation_fraction": validation_fraction,
        "seed": seed,
    }

    if train_path.is_file() and val_path.is_file() and cache_manifest.is_file():
        try:
            if json.loads(cache_manifest.read_text()) == expected:
                train_size = train_path.stat().st_size // 2
                val_size = val_path.stat().st_size // 2
                return train_size, val_size
        except (OSError, json.JSONDecodeError):
            pass

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("[EOS]")
    if eos is None or tokenizer.get_vocab_size() > 65535:
        raise ValueError("Tokenizer must contain [EOS] and have at most 65,535 tokens")

    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    train_count = val_count = 0
    train_buffer, val_buffer = [], []

    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        with corpus.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    text = json.loads(line).get("text", "")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if not isinstance(text, str) or not text:
                    continue
                ids = tokenizer.encode(text).ids + [eos]
                target = val_buffer if rng.random() < validation_fraction else train_buffer
                target.extend(ids)

                if len(train_buffer) >= 100000:
                    np.asarray(train_buffer, dtype=np.uint16).tofile(train_file)
                    train_count += len(train_buffer)
                    train_buffer.clear()
                if len(val_buffer) >= 100000:
                    np.asarray(val_buffer, dtype=np.uint16).tofile(val_file)
                    val_count += len(val_buffer)
                    val_buffer.clear()

                if line_number % 1000 == 0:
                    print(f"Tokenizing documents: {line_number:,}", flush=True)

        if train_buffer:
            np.asarray(train_buffer, dtype=np.uint16).tofile(train_file)
            train_count += len(train_buffer)
        if val_buffer:
            np.asarray(val_buffer, dtype=np.uint16).tofile(val_file)
            val_count += len(val_buffer)

    if train_count < 1000 or val_count < 100:
        raise RuntimeError(
            f"Encoded corpus too small: train={train_count}, validation={val_count}"
        )

    temp_manifest = cache_manifest.with_suffix(".tmp")
    temp_manifest.write_text(json.dumps(expected, indent=2))
    temp_manifest.replace(cache_manifest)
    return train_count, val_count


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches=20):
    model.eval()
    losses = []
    for index, (x, y) in enumerate(loader):
        if index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(output, model, optimizer, step, tokenizer_path, config):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    shutil.copy2(tokenizer_path, output / "tokenizer.json")
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config.__dict__,
    }
    temporary = output / "training_state.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "training_state.pt")
    print(f"Checkpoint saved at step {step:,}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain CrazeAI's decoder-only model from random initialization."
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/raw.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("outputs/tokenizer/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-base"))
    parser.add_argument("--steps", type=int, default=1000, help="Total target optimizer steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.block_size < 8:
        parser.error("--steps, --batch-size must be positive and --block-size >= 8")
    if args.save_interval < 1 or args.eval_interval < 1:
        parser.error("--save-interval and --eval-interval must be positive")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if not args.corpus.is_file():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}. Run scripts/train_tokenizer.py first."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
        print(f"Mixed precision: {'BF16' if use_bf16 else 'FP16'}", flush=True)
    else:
        amp_dtype = torch.float32
        scaler = None
        print("WARNING: CUDA unavailable; training on CPU will be slow.", flush=True)

    print("Encoding corpus...", flush=True)
    train_tokens, val_tokens = encode_corpus(
        args.corpus, args.tokenizer, args.output,
        args.validation_fraction, args.seed
    )
    print(
        f"Encoded tokens: train={train_tokens:,}, validation={val_tokens:,}",
        flush=True
    )

    train_data = TokenBlocks(args.output / "train.bin", args.block_size)
    val_data = TokenBlocks(args.output / "validation.bin", args.block_size)
    if len(train_data) == 0 or len(val_data) == 0:
        raise RuntimeError(
            "Not enough encoded tokens for the selected block size. "
            "Collect more data or reduce --block-size."
        )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config = CrazeAIConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=args.block_size
    )
    model = CrazeAIModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0

    state_path = args.output / "training_state.pt"
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"No resume checkpoint found at {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("config") != config.__dict__:
            raise ValueError("Checkpoint model config differs from current tokenizer/block size.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"Resumed from optimizer step {start_step:,}", flush=True)

    if start_step >= args.steps:
        print(f"Already reached requested total steps ({args.steps}). Nothing to do.", flush=True)
        return

    model.train()
    step = start_step
    while step < args.steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            step += 1
            if step == 1 or step % 10 == 0:
                print(
                    f"step {step}/{args.steps} | loss {loss.item():.4f}",
                    flush=True
                )

            if step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, amp_dtype)
                print(
                    f"validation step {step}: loss={val_loss:.4f}, "
                    f"perplexity={math.exp(min(val_loss, 20)):.2f}",
                    flush=True
                )

            if step % args.save_interval == 0 or step == args.steps:
                save_checkpoint(
                    args.output, model, optimizer, step, args.tokenizer, config
                )

            if step >= args.steps:
                break

    if step % args.save_interval != 0 and step != args.steps:
        save_checkpoint(args.output, model, optimizer, step, args.tokenizer, config)
    print(f"Pretraining finished: {step} optimizer steps.", flush=True)


if __name__ == "__main__":
    main()
