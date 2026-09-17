"""A simple character-level tokenizer for the first CrazeAI prototype."""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.itos = vocab
        self.stoi = {token: i for i, token in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi.get(" ", 0)
        return [self.stoi.get(ch, unknown) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab)
