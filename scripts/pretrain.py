from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
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
        chunk = np.asarray(self.tokens[start:start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def encode_corpus(corpus: Path, tokenizer_path: Path, output: Path, validation_fraction: float, seed: int):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("[EOS]")
    if eos is None or tokenizer.get_vocab_size() > 65535:
        raise ValueError("Tokenizer must contain [EOS] and have at most 65,535 tokens")
    rng = random.Random(seed)
    train_ids, val_ids = [], []
    with corpus.open(encoding="utf-8") as source:
        for line in source:
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not isinstance(text, str) or not text:
                continue
            ids = tokenizer.encode(text).ids + [eos]
            (val_ids if rng.random() < validation_fraction else train_ids).extend(ids)
    if len(train_ids) < 1000 or len(val_ids) < 100:
        raise RuntimeError("Corpus too small; collect more documents before pretraining.")
    output.mkdir(parents=True, exist_ok=True)
    np.asarray(train_ids, dtype=np.uint16).tofile(output / "train.bin")
    np.asarray(val_ids, dtype=np.uint16).tofile(output / "validation.bin")
    return len(train_ids), len(val_ids)


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    losses = []
    for index, (x, y) in enumerate(loader):
        if index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.reshape(-1)).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser(description="Pretrain CrazeAI from random initialization; no pretrained model is downloaded.")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/raw.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("outputs/crazeai-base/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-base"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=0, help="0 uses the exact vocabulary size of tokenizer.json")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.block_size < 8:
        parser.error("steps and batch-size must be positive; block-size must be >=8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CPU pretraining is very slow; use a CUDA GPU for meaningful runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    train_bin, val_bin = args.output / "train.bin", args.output / "validation.bin"
    if not train_bin.exists() or not val_bin.exists():
        n_train, n_val = encode_corpus(args.corpus, args.tokenizer, args.output, args.validation_fraction, args.seed)
        print(f"Encoded tokens: train={n_train:,} validation={n_val:,}")
    train_data, val_data = TokenBlocks(train_bin, args.block_size), TokenBlocks(val_bin, args.block_size)
    if len(train_data) < args.batch_size or not len(val_data):
        raise RuntimeError("Not enough token blocks for this batch/block size.")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    tokenizer_vocab_size = Tokenizer.from_file(str(args.tokenizer)).get_vocab_size()
    if args.vocab_size and args.vocab_size != tokenizer_vocab_size:
        raise ValueError(f"--vocab-size {args.vocab_size} does not match tokenizer vocabulary {tokenizer_vocab_size}")
    config = CrazeAIConfig(
        vocab_size=tokenizer_vocab_size, max_seq_len=args.block_size,
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
    )
    config.validate()
    model = CrazeAIModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    step, state_path = 0, args.output / "training_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        print(f"Resuming at step {step}")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    model.train()
    while step < args.steps:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % 25 == 0:
            print(f"step={step} train_loss={loss.item():.4f}", flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = evaluate(model, val_loader, device, 20)
            print(f"step={step} validation_loss={val_loss:.4f} perplexity={math.exp(min(val_loss, 20)):.2f}", flush=True)
        if step % args.save_interval == 0 or step == args.steps:
            model.save_pretrained(args.output)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, state_path)
    model.save_pretrained(args.output)
    print(f"Random-initialization pretraining complete: {args.output}")


if __name__ == "__main__":
    main()
