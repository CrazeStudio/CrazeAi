import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import ByteTokenizer


def iter_entries(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("entries", data.get("data", []))

    if not isinstance(data, list):
        raise ValueError("Dataset must contain a list of entries.")

    yield from data


class CrazeDataset(Dataset):
    """
    Instruction-tuning dataset.

    The loss is applied only to the assistant/output portion. This prevents
    the model from wasting training capacity learning to reproduce the prompt.
    Supports both the original JSON format and large JSONL shards.
    """

    def __init__(self, path, block_size):
        self.tok = ByteTokenizer()
        self.block = int(block_size)
        self.samples = []

        for entry in iter_entries(path):
            instruction = str(entry.get("instruction", ""))
            user_input = str(entry.get("input", ""))
            output = str(entry.get("output", ""))

            prefix = (
                f"### Instruction:\n{instruction}\n"
                f"### Input:\n{user_input}\n"
                f"### Output:\n"
            )
            full = prefix + output

            prefix_ids = self.tok.encode(prefix, add_bos=True, add_eos=False)
            full_ids = self.tok.encode(full, add_bos=True, add_eos=True)

            full_ids = full_ids[: self.block + 1]
            if len(full_ids) < 2:
                continue

            x = full_ids[:-1]
            y = full_ids[1:]

            # Ignore every target before the assistant answer begins.
            answer_start = min(max(len(prefix_ids) - 1, 0), len(y))
            targets = [-100] * len(y)
            for i in range(answer_start, len(y)):
                targets[i] = y[i]

            self.samples.append(
                (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(targets, dtype=torch.long),
                )
            )

        if not self.samples:
            raise ValueError(f"No valid training examples found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        x_pad = torch.full((self.block,), self.tok.eos, dtype=torch.long)
        y_pad = torch.full((self.block,), -100, dtype=torch.long)

        n = min(x.numel(), self.block)
        x_pad[:n] = x[:n]
        y_pad[:n] = y[:n]
        return x_pad, y_pad
