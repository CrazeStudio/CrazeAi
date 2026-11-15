from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import DownloadConfig, load_dataset


def normalize_messages(row):
    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if not isinstance(messages, list):
        return None

    normalized = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("from", ""))).strip().lower()
        content = item.get("content", item.get("value", item.get("text", "")))
        if role in {"human", "user", "prompter"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot", "chatgpt"}:
            role = "assistant"
        elif role in {"system"}:
            role = "system"
        else:
            continue
        if isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content.strip()})

    # Keep only conversations containing at least one user/assistant exchange.
    if not any(m["role"] == "user" for m in normalized):
        return None
    if not any(m["role"] == "assistant" for m in normalized):
        return None
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Prepare conversational instruction data for CrazeAI SFT."
    )
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("data/instruct/train.jsonl"))
    parser.add_argument("--min-messages", type=int, default=2)
    args = parser.parse_args()

    if args.limit < 1 or args.min_messages < 2:
        parser.error("--limit must be positive and --min-messages must be at least 2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        download_config=DownloadConfig(max_retries=10),
    )

    accepted = scanned = 0
    with args.output.open("w", encoding="utf-8") as output:
        for row in stream:
            scanned += 1
            messages = normalize_messages(row)
            if not messages or len(messages) < args.min_messages:
                continue
            output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            accepted += 1
            if accepted % 500 == 0:
                output.flush()
                print(f"accepted={accepted:,} scanned={scanned:,}", flush=True)
            if accepted >= args.limit:
                break

    if accepted == 0:
        raise RuntimeError(
            "No valid conversations found. Check the dataset name, split, and schema."
        )

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "accepted_conversations": accepted,
        "scanned_rows": scanned,
        "format": "messages[{role,content}]",
        "license_notice": (
            "Review the upstream dataset card, license, provenance, and usage terms "
            "before training or redistributing weights."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
