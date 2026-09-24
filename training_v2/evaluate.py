import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBES = [
    "Hello! Introduce yourself in one sentence.",
    "Write a Python function that reverses a string.",
    "Explain why this code raises NameError: print(username)",
    "What is HTML?",
    "Solve: If a box contains 7 red balls and 5 blue balls, how many balls are there total?",
    "Make a simple plan for a Snake game.",
    "If you do not know a current fact, what should you do?",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/crazeai-135m")
    p.add_argument("--data", default="artifacts/sft.jsonl")
    p.add_argument("--output", default="artifacts/evaluation.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()

    ds = load_dataset("json", data_files=args.data, split="train")
    test = ds.select(range(max(0, len(ds) - min(1000, len(ds)) ), len(ds))) if len(ds) else ds

    losses = []
    with torch.no_grad():
        for row in test:
            ids = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids, labels=ids)
            losses.append(float(out.loss.detach().cpu()))

    result = {
        "model": args.model,
        "eval_examples": len(losses),
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "perplexity": math.exp(sum(losses) / len(losses)) if losses else None,
        "probes": [],
    }

    for prompt in PROBES:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = text[len(prompt):].strip() if text.startswith(prompt) else text
        result["probes"].append({"prompt": prompt, "answer": answer})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
