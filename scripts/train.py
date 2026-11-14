from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from crazeai.data import CausalLMCollator, make_supervised_features, normalize_record
from crazeai.runtime import BASE_MODEL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter PyTorch fine-tuning for CrazeAI 0.5B.")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-500m"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def encode_row(row, tokenizer, max_length: int):
    messages = normalize_record(row)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}
    features = make_supervised_features(messages, tokenizer, max_length)
    if not features:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}
    return {**features, "valid": True}


def main() -> None:
    args = parse_args()
    if not args.train.is_file() or not args.validation.is_file():
        raise FileNotFoundError(
            "Training files are missing. Run scripts/prepare_dataset.py --count 50000 first."
        )
    if args.max_length < 128:
        raise ValueError("--max-length must be at least 128")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA was not detected. Full fine-tuning a 0.5B model on CPU is impractically slow. "
            "Use a CUDA GPU, or pass --allow-cpu only for a deliberate small experiment."
        )

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
        print("CUDA device:", torch.cuda.get_device_name(0))
        print("CUDA VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    else:
        torch_dtype = torch.float32
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        print("WARNING: CPU experiment selected; full training may take an impractical amount of time.")

    print("Loading tokenizer:", args.model, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading JSONL datasets", flush=True)
    raw = load_dataset(
        "json",
        data_files={"train": str(args.train), "validation": str(args.validation)},
    )
    if not len(raw["train"]) or not len(raw["validation"]):
        raise RuntimeError("Both train and validation datasets must be non-empty")

    def encode(batch):
        features = [
            encode_row(row, tokenizer, args.max_length)
            for row in zip(batch["messages"], batch.get("source", [""] * len(batch["messages"])))
        ]
        # Dataset.map with batched=True receives columns, so rebuild record-by-record.
        # The next branch handles the actual dict-of-lists form below.
        return features

    def encode_batch(batch):
        output = {"input_ids": [], "attention_mask": [], "labels": [], "valid": []}
        messages_col = batch["messages"]
        sources_col = batch.get("source", [""] * len(messages_col))
        for messages_obj, source in zip(messages_col, sources_col):
            row = {"messages": messages_obj, "source": source}
            feature = encode_row(row, tokenizer, args.max_length)
            for key in ("input_ids", "attention_mask", "labels", "valid"):
                output[key].append(feature[key])
        return output

    encoded = {}
    for split_name in ("train", "validation"):
        split = raw[split_name]
        mapped = split.map(
            encode_batch,
            batched=True,
            batch_size=128,
            num_proc=max(1, args.num_proc),
            remove_columns=split.column_names,
            desc=f"Tokenizing {split_name} with assistant-only loss",
        )
        mapped = mapped.filter(lambda row: row["valid"], desc=f"Filtering invalid {split_name} rows")
        mapped = mapped.remove_columns(["valid"])
        if len(mapped) == 0:
            raise RuntimeError(f"No valid assistant-supervised examples remain in {split_name}")
        encoded[split_name] = mapped
        print(f"{split_name}: {len(mapped):,} supervised records", flush=True)

    print("Loading model:", args.model, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Actual model parameters: {parameter_count:,} ({parameter_count / 1e9:.3f}B)")
    if not 450_000_000 <= parameter_count <= 550_000_000:
        raise RuntimeError(
            f"Loaded checkpoint is not in the requested 500M class: {parameter_count:,} parameters"
        )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    collator = CausalLMCollator(tokenizer.pad_token_id)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        dataloader_num_workers=2 if use_cuda else 0,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    trainer.save_state()

    report = {
        "project": "CrazeAI",
        "base_model": args.model,
        "actual_parameter_count": parameter_count,
        "parameter_class": "approximately 500M",
        "framework": f"PyTorch {torch.__version__}",
        "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "dtype": str(torch_dtype),
        "train_examples": len(encoded["train"]),
        "validation_examples": len(encoded["validation"]),
        "epochs_requested": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Fine-tuned from the declared pretrained checkpoint; not pretrained from random initialization.",
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
