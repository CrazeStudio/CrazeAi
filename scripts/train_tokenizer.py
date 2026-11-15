from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def main():
    parser = argparse.ArgumentParser(description="Train CrazeAI's BPE tokenizer from its collected corpus.")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/raw.jsonl"))
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-base/tokenizer.json"))
    args = parser.parse_args()
    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=True,
    )

    def text_iterator():
        with args.corpus.open(encoding="utf-8") as source:
            for line in source:
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if isinstance(text, str) and text:
                    yield text

    with args.corpus.open(encoding="utf-8") as source:
        count = sum(1 for line in source if line.strip())
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=count)
    tokenizer.save(str(args.output))
    print(f"Saved {tokenizer.get_vocab_size()} tokens to {args.output}")


if __name__ == "__main__":
    main()
