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
        torch.cuda.set_device(0)
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

    # Tokenize JSONL directly instead of Dataset.map(). In Kaggle's Python 3.13
    # environment, datasets.map fingerprints closures with dill and can emit
    # BufferError: Existing exports of data: object cannot be re-sized.
    # Direct streaming avoids dill serialization and keeps RAM usage bounded.
    print("Tokenizing JSONL directly (CPU preprocessing; no datasets.map/dill)", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenized_dir = args.output / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_jsonl(source_path: Path, destination_path: Path, split_name: str) -> int:
        valid_count = 0
        seen_count = 0
        temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
        with source_path.open("r", encoding="utf-8") as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {source_path} at line {line_number}: {exc}"
                    ) from exc
                feature = encode_row(row, tokenizer, args.max_length)
                seen_count += 1
                if feature.get("valid"):
                    feature.pop("valid", None)
                    destination.write(json.dumps(feature, separators=(",", ":")) + "\\n")
                    valid_count += 1
                if seen_count % 500 == 0:
                    print(
                        f"{split_name}: processed {seen_count:,} / "
                        f"valid {valid_count:,}",
                        flush=True,
                    )
        if valid_count == 0:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"No valid assistant-supervised examples remain in {split_name}")
        temporary_path.replace(destination_path)
        print(
            f"{split_name}: {valid_count:,} supervised records from {seen_count:,}",
            flush=True,
        )
        return valid_count

    train_tokenized = tokenized_dir / "train.jsonl"
    validation_tokenized = tokenized_dir / "validation.jsonl"
    train_count = tokenize_jsonl(args.train, train_tokenized, "train")
    validation_count = tokenize_jsonl(args.validation, validation_tokenized, "validation")

    print("Loading pre-tokenized datasets", flush=True)
    encoded = load_dataset(
        "json",
        data_files={
            "train": str(train_tokenized),
            "validation": str(validation_tokenized),
        },
    )
    if not len(encoded["train"]) or not len(encoded["validation"]):
        raise RuntimeError("Both tokenized train and validation datasets must be non-empty")
    print(
        f"Ready: {train_count:,} train / {validation_count:,} validation examples",
        flush=True,
    )

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

    # Place the model on CUDA explicitly. Tokenization above is intentionally CPU-side;
    # the model forward/backward passes and optimizer updates must run on the GPU.
    if use_cuda:
        model = model.to(torch.device("cuda:0"))
        first_parameter_device = next(model.parameters()).device
        if first_parameter_device.type != "cuda":
            raise RuntimeError(
                f"CUDA was detected but the model is on {first_parameter_device}; refusing CPU training."
            )
        print(f"Model device: {first_parameter_device}", flush=True)
        print(
            "CUDA allocated GiB before training:",
            round(torch.cuda.memory_allocated(0) / 2**30, 2),
            flush=True,
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
        use_cpu=not use_cuda,
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
    trainer_device = trainer.args.device
    print(f"Trainer device: {trainer_device}", flush=True)
    if use_cuda and trainer_device.type != "cuda":
        raise RuntimeError(
            f"CUDA is available but Hugging Face Trainer selected {trainer_device}; refusing CPU training."
        )
    if use_cuda and next(trainer.model.parameters()).device.type != "cuda":
        raise RuntimeError("Trainer model is not on CUDA; refusing to start training.")

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
