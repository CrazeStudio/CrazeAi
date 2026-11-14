from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from crazeai.data import CausalLMCollator, make_supervised_features, normalize_record
from crazeai.prompt import SYSTEM_PROMPT
from crazeai.runtime import CrazeAIRuntime


def read_validation(path: Path, tokenizer, max_length: int):
    features = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = normalize_record(record)
            if not messages:
                continue
            feature = make_supervised_features(messages, tokenizer, max_length)
            if feature:
                features.append(feature)
    if not features:
        raise RuntimeError(f"No valid validation examples found in {path}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CrazeAI checkpoint.")
    parser.add_argument("--model", default="outputs/crazeai-500m")
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-500m/evaluation.json"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not args.validation.is_file():
        raise FileNotFoundError(f"Validation dataset not found: {args.validation}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    runtime = CrazeAIRuntime(args.model)
    model = runtime.model
    tokenizer = runtime.tokenizer
    features = read_validation(args.validation, tokenizer, args.max_length)
    collator = CausalLMCollator(tokenizer.pad_token_id)
    model.eval()

    total_weighted_loss = 0.0
    total_supervised_tokens = 0
    with torch.inference_mode():
        for offset in range(0, len(features), args.batch_size):
            batch_features = features[offset : offset + args.batch_size]
            batch = collator(batch_features)
            batch = {key: value.to(runtime.device) for key, value in batch.items()}
            supervised = int((batch["labels"] != -100).sum().item())
            if supervised == 0:
                continue
            output = model(**batch)
            total_weighted_loss += float(output.loss.item()) * supervised
            total_supervised_tokens += supervised

    if total_supervised_tokens == 0:
        raise RuntimeError("Evaluation produced zero supervised tokens")
    eval_loss = total_weighted_loss / total_supervised_tokens
    perplexity = math.exp(min(eval_loss, 20.0))

    probes = [
        ("identity", "Who are you?"),
        ("arithmetic", "What is 7 multiplied by 8?"),
        ("coding", "Write a Python function that adds two integers."),
        ("uncertainty", "What should you do when you cannot verify a current fact?"),
    ]
    probe_results = []
    for name, question in probes:
        answer, prompt_tokens, completion_tokens = runtime.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_new_tokens=128,
            temperature=0.0,
        )
        probe_results.append(
            {
                "name": name,
                "prompt": question,
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "non_empty": bool(answer.strip()),
            }
        )

    report = {
        "model": args.model,
        "validation_file": str(args.validation),
        "validation_examples": len(features),
        "supervised_tokens": total_supervised_tokens,
        "token_weighted_eval_loss": eval_loss,
        "perplexity": perplexity,
        "probes": probe_results,
        "all_probes_non_empty": all(item["non_empty"] for item in probe_results),
        "note": "Probe outputs are qualitative checks, not a benchmark or a guarantee of factual accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
