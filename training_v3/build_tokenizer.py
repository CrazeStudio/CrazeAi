import argparse
import json
from pathlib import Path

import sentencepiece as spm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocab-size", type=int, default=16384)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text_file = out.with_suffix(".txt")

    with open(args.data, encoding="utf-8") as src, open(text_file, "w", encoding="utf-8") as dst:
        for line in src:
            try:
                text = json.loads(line).get("text", "").replace("\n", " ")
            except Exception:
                continue
            if text.strip():
                dst.write(text + "\n")

    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(out.with_suffix("")),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    )
    text_file.unlink(missing_ok=True)
    print(out.with_suffix(".model"))


if __name__ == "__main__":
    main()
