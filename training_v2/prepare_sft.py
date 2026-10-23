import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SYSTEM = (
    "You are CrazeAI, a concise and reliable AI assistant created by CrazeStudio. "
    "Your identity is CrazeAI. Answer the user's actual question directly and clearly. "
    "Do not repeat the prompt or invent facts. If you are unsure, say so plainly. "
    "Use readable English, complete sentences, and relevant detail."
)

SEED = [
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! I'm CrazeAI, an AI assistant created by CrazeStudio. How can I help you?"}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Write a Python function that reverses a string."}, {"role": "assistant", "content": "def reverse_text(text: str) -> str:\n    return text[::-1]\n\nThis function returns the input string with its characters in reverse order."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What should you do if you do not know a current fact?"}, {"role": "assistant", "content": "I should not guess. I should clearly say I do not know or that the information may be outdated. If a reliable search tool is available, I should verify the fact before answering."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Explain a Python NameError simply."}, {"role": "assistant", "content": "A NameError means Python tried to use a variable or function name that has not been defined. For example, print(x) raises a NameError if you have not assigned a value to x."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is HTML?"}, {"role": "assistant", "content": "HTML stands for HyperText Markup Language. It structures the content of web pages, including headings, paragraphs, links, images, forms, and sections."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is 7 multiplied by 8?"}, {"role": "assistant", "content": "7 multiplied by 8 equals 56."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Explain the difference between a bug and a feature."}, {"role": "assistant", "content": "A feature is an intended capability of a program. A bug is behavior that makes the program work incorrectly or differently from what was intended."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Give me a short study tip."}, {"role": "assistant", "content": "Study in focused 25-minute sessions, then take a short break. At the end of each session, close your notes and explain the main ideas from memory."}]},
]


def render(messages):
    parts = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content or role not in {"system", "user", "assistant"}:
            continue
        parts.append(f"<|{role}|>\n{content}\n<|end|>\n")
    return "".join(parts)


def load_craze_identity(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if isinstance(messages, list) and len(messages) >= 2:
                rows.append({"text": render(messages)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/sft.jsonl")
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.max_examples < 1:
        raise ValueError("--max-examples must be at least 1")

    random.seed(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    identity = load_craze_identity(Path(__file__).with_name("crazeai_identity.jsonl"))
    seed_rows = [{"text": render(item["messages"])} for item in SEED]
    reserved = (identity + seed_rows)[:args.max_examples]
    source_limit = max(0, args.max_examples - len(reserved))
    rows = []

    if source_limit:
        ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
        if hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=args.seed, buffer_size=max(1000, min(10000, source_limit)))
        for item in ds:
            messages = item.get("messages")
            if not isinstance(messages, list):
                continue
            normalized = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                    normalized.append({"role": role, "content": content.strip()})
            if not any(message["role"] == "user" for message in normalized):
                continue
            if not any(message["role"] == "assistant" for message in normalized):
                continue
            normalized.insert(0, {"role": "system", "content": SYSTEM})
            rows.append({"text": render(normalized)})
            if len(rows) >= source_limit:
                break

    if len(rows) < source_limit:
        raise RuntimeError(
            f"Requested {args.max_examples:,} examples but the source provided only "
            f"{len(rows) + len(reserved):,} usable examples. Refusing to silently train on fewer examples."
        )

    random.shuffle(rows)
    rows = reserved + rows
    if len(rows) != args.max_examples:
        raise RuntimeError(f"Expected exactly {args.max_examples:,} examples, produced {len(rows):,}")

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote exactly {len(rows):,} SFT examples to {out}")
    print(f"included {len(identity)} CrazeAI identity examples from the repository")
    print(f"included {min(len(seed_rows), len(reserved))} built-in answer-quality examples")


if __name__ == "__main__":
    main()
