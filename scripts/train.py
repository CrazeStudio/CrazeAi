from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# This project trains with PyTorch only. Disable optional TensorFlow/Flax
# backend discovery before importing Hugging Face libraries; it can otherwise
# emit a misleading TensorFlow CUDA-stub warning in Kaggle even when PyTorch CUDA works.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
# Avoid fast-tokenizer thread oversubscription (we parallelise with processes instead).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# Bump whenever the tokenized on-disk format or tokenization logic changes.
CACHE_FORMAT_VERSION = 2

# Sensible learning rates per mode. LoRA needs a much higher LR than full
# fine-tuning; 1e-5 with LoRA barely moves the adapters and wastes the whole run.
DEFAULT_LR_FULL = 1e-5
DEFAULT_LR_LORA = 2e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch fine-tuning (LoRA by default, optional full) for CrazeAI 0.5B.")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--model", default=BASE_MODEL_ID)
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-500m"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="0 = 2x --batch-size.")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Default: {DEFAULT_LR_LORA} for LoRA, {DEFAULT_LR_FULL} for --full-finetune.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=300)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-examples", type=int, default=0, help="Limit training to the first N tokenized examples; 0 uses the full dataset.")
    parser.add_argument("--num-proc", type=int, default=0, help="Tokenizer worker processes; 0 = auto (up to 8).")
    parser.add_argument("--dataloader-workers", type=int, default=-1, help="-1 = auto (up to 4 on CUDA).")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--full-finetune", action="store_true", help="Train every model parameter; slower and more memory-intensive than the default LoRA mode.")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: on for --full-finetune, off for LoRA (checkpointing costs ~30%% speed; enable only to save memory).",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-group-by-length", action="store_true", help="Disable length-grouped batching (more padding, slower).")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LR_FULL if args.full_finetune else DEFAULT_LR_LORA
    if args.eval_batch_size <= 0:
        args.eval_batch_size = max(1, args.batch_size * 2)
    return args


class TokenizedJsonlDataset(torch.utils.data.Dataset):
    """Random-access JSONL dataset without pandas/PyArrow JSON inference.

    On-disk records are compact: {"i": input_ids, "p": number_of_prompt_tokens}.
    Labels and the attention mask are rebuilt on the fly, which halves file size
    and JSON parsing time compared with storing three parallel lists.
    """

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
        if "i" not in feature or "p" not in feature:
            raise ValueError(f"Tokenized record {index} is missing required fields")
        input_ids = feature["i"]
        start = int(feature["p"])
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * start + input_ids[start:],
        }


def encode_compact(row, tokenizer, max_length: int):
    """Return {"i": ids, "p": prompt_len} for a valid supervised example, else None."""
    messages = normalize_record(row)
    if not messages:
        return None
    features = make_supervised_features(messages, tokenizer, max_length)
    if not features:
        return None
    labels = features["labels"]
    start = next((position for position, label in enumerate(labels) if label != -100), None)
    if start is None:
        return None
    return {"i": features["input_ids"], "p": start}


# --- multiprocessing tokenization workers (module-level so "spawn" can import them) ---
_WORKER_TOKENIZER = None
_WORKER_MAX_LENGTH = 0


def _init_worker(model_id: str, max_length: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_LENGTH
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MAX_LENGTH = max_length


def _encode_worker(item):
    line_number, line = item
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        return line_number, "error", f"Invalid JSON at line {line_number}: {exc}"
    record = encode_compact(row, _WORKER_TOKENIZER, _WORKER_MAX_LENGTH)
    if record is None:
        return line_number, "skip", None
    return line_number, "ok", json.dumps(record, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_jsonl(
    source_path: Path,
    destination_path: Path,
    split_name: str,
    model_id: str,
    max_length: int,
    num_proc: int,
) -> int:
    def numbered_lines():
        with source_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    yield line_number, line

    valid_count = 0
    seen_count = 0
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    pool = None
    try:
        if num_proc > 1:
            pool = mp.get_context("spawn").Pool(
                num_proc, initializer=_init_worker, initargs=(model_id, max_length)
            )
            results = pool.imap(_encode_worker, numbered_lines(), chunksize=64)
        else:
            _init_worker(model_id, max_length)
            results = map(_encode_worker, numbered_lines())

        with temporary_path.open("w", encoding="utf-8") as destination:
            for _line_number, status, payload in results:
                if status == "error":
                    raise ValueError(f"{source_path}: {payload}")
                seen_count += 1
                if status == "ok":
                    destination.write(payload + "\n")
                    valid_count += 1
                if seen_count % 2000 == 0:
                    print(f"{split_name}: processed {seen_count:,} / valid {valid_count:,}", flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if valid_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"No valid assistant-supervised examples remain in {split_name}")
    temporary_path.replace(destination_path)
    print(f"{split_name}: {valid_count:,} supervised records from {seen_count:,}", flush=True)
    return valid_count


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
            allocated = torch.cuda.memory_allocated(self.device_index) / 2**30
            reserved = torch.cuda.memory_reserved(self.device_index) / 2**30
            print(
                f"GPU active: step {state.global_step} | "
                f"rank device cuda:{self.device_index} | "
                f"allocated {allocated:.2f} GiB | reserved {reserved:.2f} GiB",
                flush=True,
            )
        return control


def pick_attention_implementation(use_cuda: bool, use_bf16: bool) -> str:
    """FlashAttention-2 only where it is supported (Ampere+, bf16) and installed."""
    if use_cuda and use_bf16:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


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
            # Rank 0 tokenizes while the other ranks wait at a barrier. The default
            # NCCL timeout (10 min) kills multi-GPU runs on a large first tokenization.
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(hours=2)
            )
        major, minor = torch.cuda.get_device_capability(local_rank)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # IMPORTANT: torch.cuda.is_bf16_supported() reports True on T4 (sm75) via
        # *emulation*, which is several times slower than fp16 tensor cores. Use
        # bf16 only on hardware with native support (Ampere / sm80 and newer).
        use_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
        if use_bf16:
            torch_dtype = torch.bfloat16
        elif args.full_finetune:
            # fp16 *weights* cannot be trained with the fp16 GradScaler
            # ("Attempting to unscale FP16 gradients"); keep fp32 master weights
            # and let autocast run the matmuls in fp16.
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16
        if is_main_process:
            print(
                f"CUDA devices visible: {torch.cuda.device_count()} | "
                f"distributed world size: {world_size}",
                flush=True,
            )
        print(
            f"Rank {global_rank}: {torch.cuda.get_device_name(local_rank)} "
            f"(cuda:{local_rank}, sm{major}{minor}) | VRAM GiB: "
            f"{torch.cuda.get_device_properties(local_rank).total_memory / 2**30:.2f} | "
            f"precision: {'bf16' if use_bf16 else 'fp16 autocast'}",
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
    args.output.mkdir(parents=True, exist_ok=True)
    tokenized_dir = args.output / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    train_tokenized = tokenized_dir / "train.jsonl"
    validation_tokenized = tokenized_dir / "validation.jsonl"
    meta_path = tokenized_dir / "meta.json"

    if is_main_process:
        expected_meta = {
            "format": CACHE_FORMAT_VERSION,
            "model": args.model,
            "max_length": args.max_length,
            "train_sha256": file_sha256(args.train),
            "validation_sha256": file_sha256(args.validation),
        }
        cache_ok = False
        if train_tokenized.is_file() and validation_tokenized.is_file() and meta_path.is_file():
            try:
                cache_ok = json.loads(meta_path.read_text(encoding="utf-8")) == expected_meta
            except (OSError, json.JSONDecodeError):
                cache_ok = False
        if cache_ok:
            print("Reusing cached tokenized data (inputs, model and max-length unchanged)", flush=True)
        else:
            num_proc = args.num_proc if args.num_proc > 0 else max(1, min(8, os.cpu_count() or 1))
            print(f"Tokenizing JSONL with {num_proc} worker process(es) (rank 0 only; no datasets.map/dill)", flush=True)
            meta_path.unlink(missing_ok=True)
            tokenize_jsonl(args.train, train_tokenized, "train", args.model, args.max_length, num_proc)
            tokenize_jsonl(args.validation, validation_tokenized, "validation", args.model, args.max_length, num_proc)
            meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")
    if distributed:
        torch.distributed.barrier(device_ids=[local_rank])
    if not train_tokenized.is_file() or not validation_tokenized.is_file():
        raise FileNotFoundError("Rank 0 did not produce the shared tokenized JSONL files.")

    if is_main_process:
        print("Loading pre-tokenized JSONL with native PyTorch reader (no pandas/PyArrow)", flush=True)
    train_dataset = TokenizedJsonlDataset(train_tokenized)
    validation_dataset = TokenizedJsonlDataset(validation_tokenized)
    if args.max_train_examples < 0:
        raise ValueError("--max-train-examples must be 0 (unlimited) or a positive integer")
    if args.max_train_examples:
        train_dataset.offsets = train_dataset.offsets[:args.max_train_examples]
        if not train_dataset.offsets:
            raise ValueError("--max-train-examples selected zero training examples")
        if is_main_process:
            print(
                f"Development limit enabled: using {len(train_dataset):,} training examples "
                f"(requested maximum {args.max_train_examples:,})",
                flush=True,
            )
    if is_main_process:
        print(
            f"Ready: {len(train_dataset):,} train / "
            f"{len(validation_dataset):,} validation examples",
            flush=True,
        )

    if is_main_process:
        print("Loading model:", args.model, flush=True)
    attn_implementation = pick_attention_implementation(use_cuda, use_bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        trust_remote_code=False,
    )
    if is_main_process:
        print(f"Attention implementation: {attn_implementation}", flush=True)
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
    use_gradient_checkpointing = (
        args.full_finetune if args.gradient_checkpointing is None else args.gradient_checkpointing
    )
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

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
        # The fp16 GradScaler requires fp32 trainable parameters.
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype == torch.float16:
                parameter.data = parameter.data.float()
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.print_trainable_parameters()
        print(f"Training mode: LoRA | trainable parameters: {trainable_parameters:,}", flush=True)
    else:
        print("Training mode: full fine-tuning (slower)", flush=True)
    if is_main_process:
        print(
            f"Gradient checkpointing: {use_gradient_checkpointing} | learning rate: {args.learning_rate:g} | "
            f"effective batch: {args.batch_size * args.gradient_accumulation * world_size}",
            flush=True,
        )

    if args.dataloader_workers >= 0:
        loader_workers = args.dataloader_workers
    else:
        loader_workers = min(4, os.cpu_count() or 1) if use_cuda else 0

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
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Batch similar-length examples together to cut padding waste.
        group_by_length=not args.no_group_by_length,
        # Fused AdamW is noticeably faster than the default on CUDA.
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        dataloader_num_workers=loader_workers,
        dataloader_persistent_workers=loader_workers > 0,
        dataloader_prefetch_factor=4 if loader_workers > 0 else None,
        dataloader_pin_memory=use_cuda,
        ddp_backend="nccl" if distributed else None,
        ddp_find_unused_parameters=False if distributed else None,
        ddp_timeout=7200,
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
        "mixed_precision": "bf16" if use_bf16 else ("fp16" if use_cuda else "none"),
        "attention": attn_implementation,
        "gradient_checkpointing": use_gradient_checkpointing,
        "group_by_length": not args.no_group_by_length,
        "train_examples": len(train_dataset),
        "max_train_examples": args.max_train_examples or None,
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
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
