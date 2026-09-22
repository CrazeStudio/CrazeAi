import json
import torch
from torch.utils.data import Dataset
from tokenizer import ByteTokenizer

class CrazeDataset(Dataset):
    def __init__(self, path, block_size):
        self.tok = ByteTokenizer()
        self.block = block_size
        with open(path, encoding="utf-8") as f:
            data = json.load(f)["entries"]
        self.samples = []
        for e in data:
            text = (
                f"### Instruction:\n{e.get('instruction', '')}\n"
                f"### Input:\n{e.get('input', '')}\n"
                f"### Output:\n{e.get('output', '')}"
            )
            ids = self.tok.encode(text)
            ids = ids[:block_size + 1]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids = self.samples[i]
        x = ids[:-1]
        y = ids[1:]
        x_pad = torch.zeros(self.block, dtype=torch.long)
        y_pad = torch.full((self.block,), -100, dtype=torch.long)
        n = min(len(x), self.block)
        x_pad[:n] = x[:n]
        y_pad[:n] = y[:n]
        return x_pad, y_pad
