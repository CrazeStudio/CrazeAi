from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

# This project trains with PyTorch only. Disable optional TensorFlow/Flax
# backend discovery before importing Hugging Face libraries; it can otherwise
# emit a misleading TensorFlow CUDA-stub warning in Kaggle even when PyTorch CUDA works.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
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
    parser.add_argument("--full-finetune", action="store_true", help="Train every model parameter; slower and more memory-intensive than the default LoRA mode.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class TokenizedJsonlDataset(torch.utils.data.Dataset):
    """Random-access JSONL dataset without pandas/PyArrow JSON inference."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as source:
            for line in source:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if not self.offsets:
            raise RuntimeError(f"No tokenized examples found in {self.path}")
        self._file = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = self.path.open("rb")
            self._pid = pid
        self._file.seek(self.offsets[index])
        raw = self._file.readline()
        try:
            feature = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Invalid tokenized JSONL record {index} in {self.path}"
            ) from exc
        required = ("input_ids", "attention_mask", "labels")
        if any(key not in feature for key in required):
            raise ValueError(f"Tokenized record {index} is missing required fields")
        return {key: feature[key] for key in required}


def encode_row(row, tokenizer, max_length: int):
    messages = normalize_record(row)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}
    features = make_supervised_features(messages, tokenizer, max_length)
    if not features:
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}
    return {**features, "valid": True}


class CudaStepMonitor(TrainerCallback):
    """Report real CUDA allocation during training, not just CUDA availability."""

    def __init__(self, interval: int, device_index: int, is_main_process: bool):
        self.interval = max(1, interval)
        self.device_index = device_index
        self.is_main_process = is_main_process

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return control
        if self.is_main_process and (
            state.global_step == 1 or state.global_step % self.interval == 0
        ):
            torch.cuda.synchronize(self.device_index)
            allocated = torch.cuda.memory_allocated(self.device_index) / 2**30
            reserved = torch.cuda.memory_reserved(self.device_index) / 2**30
            print(
                f"GPU active: step {state.global_step} | "
                f"rank device cuda:{self.device_index} | "
                f"allocated {allocated:.2f} GiB | reserved {reserved:.2f} GiB",
                flush=True,
            )
        return control


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
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    is_main_process = global_rank == 0
    use_bf16 = False

    if distributed and not use_cuda:
        raise RuntimeError("Multi-process DDP requires CUDA GPUs; CUDA was not detected.")
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(local_rank)
        if distributed and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_bf16 = torch.cuda.is_bf16_supported()
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
        if is_main_process:
            print(
                f"CUDA devices visible: {torch.cuda.device_count()} | "
                f"distributed world size: {world_size}",
                flush=True,
            )
        print(
            f"Rank {global_rank}: {torch.cuda.get_device_name(local_rank)} "
            f"(cuda:{local_rank}) | VRAM GiB: "
            f"{torch.cuda.get_device_properties(local_rank).total_memory / 2**30:.2f}",
            flush=True,
        )
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
    if is_main_process:
        print("Tokenizing JSONL directly (CPU preprocessing; rank 0 only; no datasets.map/dill)", flush=True)
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
                    destination.write(json.dumps(feature, separators=(",", ":")) + "\n")
                    valid_count += 1
                if is_main_process and seen_count % 500 == 0:
                    print(
                        f"{split_name}: processed {seen_count:,} / "
                        f"valid {valid_count:,}",
                        flush=True,
                    )
        if valid_count == 0:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"No valid assistant-supervised examples remain in {split_name}")
        temporary_path.replace(destination_path)
        if is_main_process:
            print(
                f"{split_name}: {valid_count:,} supervised records from {seen_count:,}",
                flush=True,
            )
        return valid_count

    train_tokenized = tokenized_dir / "train.jsonl"
    validation_tokenized = tokenized_dir / "validation.jsonl"
    if is_main_process:
        tokenize_jsonl(args.train, train_tokenized, "train")
        tokenize_jsonl(args.validation, validation_tokenized, "validation")
    if distributed:
        torch.distributed.barrier()
    if not train_tokenized.is_file() or not validation_tokenized.is_file():
        raise FileNotFoundError("Rank 0 did not produce the shared tokenized JSONL files.")

    if is_main_process:
        print("Loading pre-tokenized JSONL with native PyTorch reader (no pandas/PyArrow)", flush=True)
    train_dataset = TokenizedJsonlDataset(train_tokenized)
    validation_dataset = TokenizedJsonlDataset(validation_tokenized)
    if is_main_process:
        print(
            f"Ready: {len(train_dataset):,} train / "
            f"{len(validation_dataset):,} validation examples",
            flush=True,
        )

    if is_main_process:
        print("Loading model:", args.model, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if is_main_process:
        print(f"Actual model parameters: {parameter_count:,} ({parameter_count / 1e9:.3f}B)")
    if not 450_000_000 <= parameter_count <= 550_000_000:
        raise RuntimeError(
            f"Loaded checkpoint is not in the requested 500M class: {parameter_count:,} parameters"
        )

    # Place the model on CUDA explicitly. Tokenization above is intentionally CPU-side;
    # the model forward/backward passes and optimizer updates must run on the GPU.
    if use_cuda:
        model = model.to(torch.device(f"cuda:{local_rank}"))
        first_parameter_device = next(model.parameters()).device
        if first_parameter_device.type != "cuda":
            raise RuntimeError(
                f"CUDA was detected but the model is on {first_parameter_device}; refusing CPU training."
            )
        print(f"Rank {global_rank} model device: {first_parameter_device}", flush=True)
        print(
            f"Rank {global_rank} CUDA allocated GiB before training:",
            round(torch.cuda.memory_allocated(local_rank) / 2**30, 2),
            flush=True,
        )

    model.config.use_cache = False
    trainable_parameters = parameter_count
    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.print_trainable_parameters()
        print(f"Training mode: LoRA | trainable parameters: {trainable_parameters:,}", flush=True)
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("Training mode: full fine-tuning (slower)", flush=True)


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
        gradient_checkpointing=args.full_finetune,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        dataloader_num_workers=2 if use_cuda else 0,
        dataloader_pin_memory=use_cuda,
        ddp_backend="nccl" if distributed else None,
        ddp_find_unused_parameters=False if distributed else None,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[CudaStepMonitor(args.logging_steps, local_rank, is_main_process)],
    )
    trainer_device = trainer.args.device
    print(f"Rank {global_rank} trainer device: {trainer_device}", flush=True)
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
    if is_main_process:
        tokenizer.save_pretrained(str(args.output))
        trainer.save_state()

    report = {
        "project": "CrazeAI",
        "base_model": args.model,
        "actual_parameter_count": parameter_count,
        "training_mode": "full-finetune" if args.full_finetune else "LoRA",
        "trainable_parameter_count": trainable_parameters,
        "lora_rank": None if args.full_finetune else args.lora_r,
        "parameter_class": "approximately 500M",
        "framework": f"PyTorch {torch.__version__}",
        "device": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if use_cuda else "CPU"
        ),
        "world_size": world_size,
        "dtype": str(torch_dtype),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
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
    if is_main_process:
        (args.output / "training_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, default=str))

    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
