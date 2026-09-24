import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SYSTEM = (
    "You are CrazeAI, a concise and reliable AI assistant created by CrazeStudio. "
    "Your identity is CrazeAI. Do not claim to be Qwen, SmolLM, ChatGPT, or another assistant. "
    "Answer directly, reason carefully, and never invent facts. If information is uncertain, say so. "
    "Current or time-sensitive facts must be obtained with the web tool when available."
)

SEED = [
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Write a Python function that reverses a string."}, {"role": "assistant", "content": "```python\ndef reverse_text(text: str) -> str:\n    return text[::-1]\n```\n\nThis returns the input string in reverse order."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What should you do if you do not know a current fact?"}, {"role": "assistant", "content": "I should not guess. I should say that the information may be unknown or outdated and, when a web-search tool is available, use it to verify the current fact."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Explain a Python NameError simply."}, {"role": "assistant", "content": "A NameError means Python tried to use a name, such as a variable or function, that has not been defined in the current scope. For example, `print(x)` raises NameError if `x` was never created."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Make a small snake game."}, {"role": "assistant", "content": "A good first implementation is a grid-based game with a snake list, a direction vector, food coordinates, collision checks, and a fixed update loop. I would keep rendering and game-state logic separate so the game is easy to extend."}]},
    {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is HTML?"}, {"role": "assistant", "content": "HTML (HyperText Markup Language) is the markup language used to structure content in web pages, such as headings, paragraphs, links, images, forms, and sections."}]},
]


def render(messages):
    parts = []
    for m in messages:
        role = m.get("role", "").strip().lower()
        content = str(m.get("content", "")).strip()
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
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if isinstance(messages, list) and len(messages) >= 2:
                rows.append({"text": render(messages)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/sft.jsonl")
    p.add_argument("--max-examples", type=int, default=250000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    rows = []
    for item in ds:
        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        normalized = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content.strip()})
        if len(normalized) >= 2 and any(x["role"] == "assistant" for x in normalized):
            rows.append({"text": render(normalized)})
        if len(rows) >= args.max_examples:
            break

    random.shuffle(rows)
    identity = load_craze_identity(Path(__file__).with_name("crazeai_identity.jsonl"))
    seed_rows = [{"text": render(x["messages"])} for x in SEED]
    # Identity/capability examples are always placed first so the model sees them
    # deterministically in every training run.
    rows = identity + seed_rows + rows

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows):,} SFT examples to {out}")
    print(f"included {len(identity)} CrazeAI identity examples")


if __name__ == "__main__":
    main()
