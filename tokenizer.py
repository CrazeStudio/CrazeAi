import json


class CharTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.itos = {int(v): k for k, v in self.vocab.items()}

    def train(self, text):
        chars = sorted(set(text))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.vocab.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        if not self.vocab:
            raise RuntimeError("Tokenizer has not been trained")
        return [self.vocab.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))
