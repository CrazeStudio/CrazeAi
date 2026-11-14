from __future__ import annotations

from typing import Any

from .prompt import SYSTEM_PROMPT

_ALLOWED_ROLES = {"system", "user", "assistant"}
_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def normalize_record(record: Any) -> list[dict[str, str]] | None:
    """Normalize supported dataset schemas into a bounded chat conversation."""
    if not isinstance(record, dict):
        return None

    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        instruction = record.get("instruction") or record.get("prompt")
        response = record.get("output") or record.get("response") or record.get("completion")
        extra_input = record.get("input")
        if not isinstance(instruction, str) or not isinstance(response, str):
            return None
        user_text = instruction.strip()
        if isinstance(extra_input, str) and extra_input.strip():
            user_text += "\n\n" + extra_input.strip()
        raw_messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response.strip()},
        ]

    messages: list[dict[str, str]] = []
    for item in raw_messages[:64]:
        if not isinstance(item, dict):
            continue
        role = _ROLE_ALIASES.get(str(item.get("role", "")).strip().lower())
        content = item.get("content")
        if role not in _ALLOWED_ROLES or not isinstance(content, str):
            continue
        content = content.strip()
        if not content or len(content) > 32_000:
            continue
        messages.append({"role": role, "content": content})

    if not any(m["role"] == "user" for m in messages):
        return None
    assistant_positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_positions:
        return None

    # Supervise the final assistant turn only. Earlier turns remain context.
    messages = messages[: assistant_positions[-1] + 1]
    if not any(m["role"] == "user" for m in messages[:-1]):
        return None
    if messages[-1]["role"] != "assistant":
        return None
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def _as_list(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def make_supervised_features(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]] | None:
    """Build causal-LM labels that train on the final assistant answer only."""
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not messages or messages[-1].get("role") != "assistant":
        return None

    prompt_messages = messages[:-1]
    if not any(message.get("role") == "user" for message in prompt_messages):
        return None

    prompt_ids = _as_list(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _as_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        return None
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer chat template does not preserve the prompt prefix; "
            "refusing to construct incorrectly masked training labels."
        )

    # Keep the most recent tokens if an example exceeds the context budget.
    left_cut = max(0, len(full_ids) - max_length)
    input_ids = full_ids[left_cut:]
    supervised_start = max(0, len(prompt_ids) - left_cut)
    labels = ([-100] * supervised_start) + input_ids[supervised_start:]
    if not any(token_id != -100 for token_id in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class CausalLMCollator:
    """Right-pad tokenized rows and ignore all padding in the loss."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            if length == 0:
                raise ValueError("Encountered an empty token sequence")
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = torch.tensor(
                feature["attention_mask"], dtype=torch.long
            )
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
