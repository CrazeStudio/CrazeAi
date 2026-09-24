import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE = "HuggingFaceTB/SmolLM2-135M"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/sft.jsonl")
    p.add_argument("--output", default="artifacts/crazeai-135m")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        batch_size = args.batch_size or 8
        grad_accum = args.grad_accum or 4
        dtype = torch.bfloat16
    else:
        # GitHub-hosted runners do not provide CUDA. Keep the CPU path small
        # enough to finish instead of spending hours on a full SFT run.
        batch_size = args.batch_size or 1
        grad_accum = args.grad_accum or 16
        dtype = torch.float32
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        print("CUDA unavailable: using CPU SFT fallback")

    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=args.data, split="train")
    split = raw.train_test_split(test_size=min(2000, max(1, len(raw) // 20)), seed=42)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = split.map(
        tokenize,
        batched=True,
        remove_columns=split["train"].column_names,
        desc="Tokenizing CrazeAI SFT data",
    )

    # Use the current Transformers argument names that are broadly compatible
    # with the versions available on GitHub-hosted runners.
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
    model.config.use_cache = False
    if use_cuda:
        model.gradient_checkpointing_enable()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    training = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        # Use warmup_steps rather than warmup_ratio because some runner
        # environments expose an older TrainingArguments API.
        warmup_steps=50,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        tf32=use_cuda,
        gradient_checkpointing=use_cuda,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0 if not use_cuda else 2,
    )

    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )
    trainer.train()
    metrics = trainer.evaluate()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
