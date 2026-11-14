from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from crazeai.data import normalize_record

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "database.jsonl"
SEED_FILE = ROOT / "data" / "crazeai_seed.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed"

# All three selected SmolTalk subsets are newly created Apache-2.0 subsets.
# Counts sum to 49,988; 12 local CrazeAI seed examples complete the 50K corpus.
HF_DATASET = "HuggingFaceTB/smoltalk"
HF_MIX = (
    ("smol-magpie-ultra", 31_992),
    ("smol-rewrite", 8_998),
    ("smol-summarize", 8_998),
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                continue
            yield value


def canonical_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def load_seed_rows() -> list[tuple[list[dict[str, str]], str]]:
    if not SEED_FILE.is_file():
        raise FileNotFoundError(f"Required CrazeAI seed file is missing: {SEED_FILE}")
    rows = []
    for record in iter_jsonl(SEED_FILE):
        messages = normalize_record(record)
        if messages:
            rows.append((messages, "crazeai-seed"))
    if not rows:
        raise RuntimeError("CrazeAI seed file contains no valid conversations")
    return rows


def load_local_rows(
    path: Path,
    needed: int,
    seed: int,
    reserved_hashes: set[str] | None = None,
) -> list[tuple[list[dict[str, str]], str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Local database was requested but not found: {path}")
    rng = random.Random(seed)
    selected: list[tuple[list[dict[str, str]], str]] = []
    seen: set[str] = set(reserved_hashes or ())
    unique_seen = 0

    # Reservoir sampling keeps a deterministic representative subset without
    # loading a potentially large local database fully into memory.
    for record in iter_jsonl(path):
        messages = normalize_record(record)
        if not messages:
            continue
        key = canonical_hash(messages)
        if key in seen:
            continue
        seen.add(key)
        unique_seen += 1
        item = (messages, "local-database")
        if len(selected) < needed:
            selected.append(item)
        else:
            slot = rng.randrange(unique_seen)
            if slot < needed:
                selected[slot] = item

    if len(selected) < needed:
        raise RuntimeError(
            f"Local database has only {len(selected):,} unique usable records; "
            f"{needed:,} are required after adding CrazeAI seed examples. "
            "No remote fallback was used."
        )
    rng.shuffle(selected)
    return selected


def load_huggingface_rows(
    needed: int,
    seed: int,
    reserved_hashes: set[str] | None = None,
) -> list[tuple[list[dict[str, str]], str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required. Install requirements.txt before preparation."
        ) from exc

    selected: list[tuple[list[dict[str, str]], str]] = []
    seen: set[str] = set(reserved_hashes or ())
    for config_name, quota in HF_MIX:
        remaining = min(quota, needed - len(selected))
        if remaining <= 0:
            break
        print(f"Streaming {config_name}: target {remaining:,} unique records", flush=True)
        stream = load_dataset(
            HF_DATASET,
            config_name,
            split="train",
            streaming=True,
        ).shuffle(seed=seed, buffer_size=20_000)

        accepted = 0
        for record in stream:
            messages = normalize_record(record)
            if not messages:
                continue
            key = canonical_hash(messages)
            if key in seen:
                continue
            seen.add(key)
            selected.append((messages, config_name))
            accepted += 1
            if accepted >= remaining:
                break

        if accepted < remaining:
            raise RuntimeError(
                f"Dataset subset {config_name!r} yielded {accepted:,} unique usable records; "
                f"{remaining:,} were required. Preparation stopped."
            )

    if len(selected) != needed:
        raise RuntimeError(
            f"Expected {needed:,} remote records but selected {len(selected):,}."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic CrazeAI 50K SFT corpus.")
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--source", choices=("auto", "local", "hf"), default="auto")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count < 100:
        raise ValueError("--count must be at least 100")
    if not 0.01 <= args.validation_ratio <= 0.2:
        raise ValueError("--validation-ratio must be between 0.01 and 0.20")

    seeds = load_seed_rows()
    if len(seeds) > args.count:
        raise RuntimeError("CrazeAI seed examples exceed the requested corpus size")
    needed = args.count - len(seeds)
    seed_hashes = {canonical_hash(messages) for messages, _ in seeds}

    use_local = args.source == "local" or (args.source == "auto" and args.input.is_file())
    if use_local:
        print(f"Using local database: {args.input}", flush=True)
        data_rows = load_local_rows(args.input, needed, args.seed, seed_hashes)
        source_mode = "local"
    else:
        if args.source == "local":
            raise FileNotFoundError(f"Local database was requested but not found: {args.input}")
        print(
            f"No local database selected; using {HF_DATASET} configured subsets. "
            "This is a public fallback, not a recovered private database.",
            flush=True,
        )
        data_rows = load_huggingface_rows(needed, args.seed, seed_hashes)
        source_mode = "huggingface"

    rows = seeds + data_rows
    if len(rows) != args.count:
        raise RuntimeError(f"Internal count check failed: expected {args.count}, got {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    manifest_counts = Counter(source for _, source in rows)
    serialized = [
        {"messages": messages, "source": source}
        for messages, source in rows
    ]

    validation_count = max(1, round(args.count * args.validation_ratio))
    train_count = args.count - validation_count
    train_rows = serialized[:train_count]
    validation_rows = serialized[train_count:]

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    train_hash = write_jsonl(train_path, train_rows)
    validation_hash = write_jsonl(validation_path, validation_rows)

    manifest = {
        "project": "CrazeAI",
        "dataset_id": HF_DATASET if source_mode == "huggingface" else str(args.input),
        "source_mode": source_mode,
        "requested_examples": args.count,
        "unique_examples": len(serialized),
        "duplicate_policy": "SHA-256 over normalized message JSON; duplicates rejected",
        "seed": args.seed,
        "mix_counts": dict(sorted(manifest_counts.items())),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "validation_ratio": args.validation_ratio,
        "training_file": train_path.name,
        "training_sha256": train_hash,
        "validation_file": validation_path.name,
        "validation_sha256": validation_hash,
        "hf_configs": list(HF_MIX) if source_mode == "huggingface" else [],
        "license_note": (
            "The selected SmolTalk subsets are described as Apache-2.0 on the upstream dataset card. "
            "For local data, the owner must verify each source license and usage rights."
            if source_mode == "huggingface"
            else "Local dataset licensing and provenance were not verified by this script."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Prepared exactly {args.count:,} unique examples.")


if __name__ == "__main__":
    main()
