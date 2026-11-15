from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Stream and filter a public text corpus for CrazeAI pretraining.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/corpus/raw.jsonl"))
    args = parser.parse_args()
    if args.limit < 1 or args.min_chars < 1:
        parser.error("--limit and --min-chars must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
    seen, accepted, scanned = set(), 0, 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in stream:
            scanned += 1
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            text = " ".join(text.split())
            if len(text) < args.min_chars:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            out.write(json.dumps({"text": text, "sha256": digest}, ensure_ascii=False) + "\n")
            accepted += 1
            if accepted % 1000 == 0:
                print(f"accepted={accepted:,} scanned={scanned:,}", flush=True)
            if accepted >= args.limit:
                break
    if not accepted:
        raise RuntimeError("No documents accepted; verify dataset/config access.")
    manifest = {
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "accepted_documents": accepted, "scanned_documents": scanned,
        "min_chars": args.min_chars,
        "license_notice": "Review the upstream dataset card, source provenance, and license before training or redistributing weights.",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
