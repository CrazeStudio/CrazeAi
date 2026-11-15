from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split

from crazeai.model import CrazeAIModel


def load_sft_examples(dataset_path: Path, tokenizer: Tokenizer, block_size: int):
    eos_id = tokenizer.token_to_id("[EOS]")
    bos_id = tokenizer.token_to_id("[BOS]")
    if eos_id is None:
        raise ValueError("Tokenizer is missing the [EOS] token")

    examples = []
    skipped = 0
    with dataset_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = row.get("messages")
            if not isinstance(messages, list):
                skipped += 1
                continue

            history = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "")).lower()
                content = message.get("content", "")
                if role not in {"user", "assistant", "system"}:
                    continue
                if not isinstance(content, str) or not content.strip():
                    continue
                content = content.strip()

                if role == "assistant":
                    prompt = "\n".join(history) + "\nASSISTANT:"
                    prompt_ids = tokenizer.encode(prompt).ids
                    if bos_id is not None:
                        prompt_ids = [bos_id] + prompt_ids
                    answer_ids = tokenizer.encode(" " + content).ids
                    answer_ids.append(eos_id)

                    # Keep the answer intact when possible; trim old prompt context first.
                    max_tokens = block_size + 1
                    if len(answer_ids) >= max_tokens:
                        answer_ids = answer_ids[:max_tokens - 1] + [eos_id]
                        prompt_ids = [bos_id] if bos_id is not None else []
                    else:
                        prompt_budget = max_tokens - len(answer_ids)
                        if len(prompt_ids) > prompt_budget:
                            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []

                    sequence = prompt_ids + answer_ids
                    if len(sequence) < 2:
                        skipped += 1
                    else:
                        labels = sequence[1:].copy()
                        prompt_label_count = max(0, len(prompt_ids) - 1)
                        labels[:prompt_label_count] = [-100] * prompt_label_count
                        examples.append((sequence[:-1], labels))

                    history.append("ASSISTANT: " + content)
                else:
                    history.append(role.upper() + ": " + content)

    if not examples:
        raise RuntimeError(
            "No assistant training examples were produced. "
            "Expected JSONL rows with messages containing user and assistant roles."
        )
    print(
        f"Prepared {len(examples):,} assistant-turn examples; skipped {skipped:,} invalid rows.",
        flush=True,
    )
    return examples


class SFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def make_collate(pad_id: int):
    def collate(batch):
        max_length = max(len(item[0]) for item in batch)
        input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
        for index, (tokens, targets) in enumerate(batch):
            length = len(tokens)
            input_ids[index, :length] = torch.tensor(tokens, dtype=torch.long)
            labels[index, :length] = torch.tensor(targets, dtype=torch.long)
        return input_ids, labels
    return collate


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches=30):
    model.eval()
    losses = []
    for index, (input_ids, labels) in enumerate(loader):
        if index >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(output, model, optimizer, step, tokenizer_path, config):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    destination_tokenizer = output / "tokenizer.json"
    if tokenizer_path.resolve() != destination_tokenizer.resolve():
        shutil.copy2(tokenizer_path, destination_tokenizer)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config,
    }
    temporary = output / "training_state.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "training_state.pt")
    print(f"SFT checkpoint saved at step {step:,}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for CrazeAI with assistant-only loss."
    )
    parser.add_argument("--base-model", type=Path, default=Path("outputs/crazeai-base"))
    parser.add_argument("--dataset", type=Path, default=Path("data/instruct/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-instruct"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.block_size < 8 or args.save_interval < 1 or args.eval_interval < 1:
        parser.error("--block-size must be >=8 and intervals must be positive")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if not (args.base_model / "config.json").is_file():
        raise FileNotFoundError(f"Base model config not found: {args.base_model}")
    if not (args.base_model / "model.pt").is_file():
        raise FileNotFoundError(f"Base model weights not found: {args.base_model / 'model.pt'}")
    if not (args.base_model / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Base tokenizer not found: {args.base_model / 'tokenizer.json'}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {args.dataset}")

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
        scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
        print(f"Mixed precision: {'BF16' if use_bf16 else 'FP16'}", flush=True)
    else:
        amp_dtype = torch.float32
        scaler = None
        print("WARNING: CUDA unavailable; SFT on CPU will be slow.", flush=True)

    tokenizer_path = args.base_model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise ValueError("Tokenizer is missing [PAD]")
    base_model = CrazeAIModel.from_pretrained_dir(args.base_model, device)
    if tokenizer.get_vocab_size() != base_model.config.vocab_size:
        raise ValueError("Base tokenizer vocabulary does not match model config")
    if args.block_size > base_model.config.max_seq_len:
        raise ValueError(
            f"--block-size {args.block_size} exceeds base model context "
            f"{base_model.config.max_seq_len}"
        )

    print("Preparing assistant-only SFT examples...", flush=True)
    examples = load_sft_examples(args.dataset, tokenizer, args.block_size)
    if len(examples) < 2:
        raise RuntimeError("At least two valid assistant-turn examples are required.")

    generator = torch.Generator().manual_seed(args.seed)
    val_count = max(1, int(len(examples) * args.validation_fraction))
    train_count = len(examples) - val_count
    train_set, val_set = random_split(
        SFTDataset(examples), [train_count, val_count], generator=generator
    )
    collate = make_collate(pad_id)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=collate
    )

    model = base_model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config_signature = {
        "vocab_size": model.config.vocab_size,
        "max_seq_len": model.config.max_seq_len,
        "d_model": model.config.d_model,
        "n_layers": model.config.n_layers,
        "n_heads": model.config.n_heads,
    }
    start_step = 0
    state_path = args.output / "training_state.pt"

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"No SFT resume checkpoint found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config") != config_signature:
            raise ValueError("SFT checkpoint architecture differs from the base model.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state"]]
            )
        print(f"Resumed SFT from step {start_step:,}", flush=True)

    if start_step >= args.steps:
        print(f"Already reached requested total SFT steps ({args.steps}).", flush=True)
        return

    model.train()
    step = start_step
    while step < args.steps:
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
            else:
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SFT loss at step {step + 1}")
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
                print(f"SFT step {step}/{args.steps} | loss {loss.item():.4f}", flush=True)

            if step % args.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, amp_dtype)
                print(
                    f"SFT validation step {step}: loss={val_loss:.4f}, "
                    f"perplexity={math.exp(min(val_loss, 20)):.2f}",
                    flush=True,
                )

            if step % args.save_interval == 0 or step == args.steps:
                save_checkpoint(
                    args.output, model, optimizer, step, tokenizer_path, config_signature
                )

            if step >= args.steps:
                break

    if step % args.save_interval != 0 and step != args.steps:
        save_checkpoint(
            args.output, model, optimizer, step, tokenizer_path, config_signature
        )
    print(f"SFT finished: {step} optimizer steps. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
