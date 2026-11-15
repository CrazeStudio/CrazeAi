from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from crazeai.model import CrazeAIConfig, CrazeAIModel


class TokenBlocks(Dataset):
    def __init__(self, path: Path, block_size: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.count = max(0, (len(self.tokens) - 1) // block_size)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = np.asarray(self.tokens[start:start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def encode_corpus(corpus: Path, tokenizer_path: Path, output: Path, validation_fraction: float, seed: int):
    import hashlib

    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()
    cache_manifest = output / "encoded_manifest.json"
    train_path, val_path = output / "train.bin", output / "validation.bin"
    corpus_stat = corpus.stat()
    expected = {
        "tokenizer_sha256": tokenizer_hash,
        "corpus_size": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    if train_path.is_file() and val_path.is_file() and cache_manifest.is_file():
        if json.loads(cache_manifest.read_text()) == expected:
            return int(np.memmap(train_path, dtype=np.uint16, mode="r").size), int(np.memmap(val_path, dtype=np.uint16, mode="r").size)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("[EOS]")
    if eos is None or tokenizer.get_vocab_size() > 65535:
        raise ValueError("Tokenizer must contain [EOS] and have at most 65,535 tokens")
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    train_count = val_count = 0
    train_buffer, val_buffer = [], []
    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        with corpus.open(encoding="utf-8") as source:
            for line in source:
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if not isinstance(text, str) or not text:
                    continue
                ids = tokenizer.encode(text).ids + [eos]
                target = val_buffer if rng.random() < validation_fraction else train_buffer
                target.extend(ids)
                if len(train_buffer) >= 100000:
                    np.asarray(train_buffer, dtype=np.uint16).tofile(train_file)
                    train_count += len(train_buffer)
                    train_buffer.clear()
                if len(val_buffer) >= 100000:
                    np.asarray(val_buffer, dtype=np.uint16).tofile(val_file)
                    val_count += len(val_buffer)
                    val_buffer.clear()
        if train_buffer:
            np.asarray(train_buffer, dtype=np.uint16).tofile(train_file)
            train_count += len(train_buffer)
        if val_buffer:
            np.asarray(val_buffer, dtype=np.uint16).tofile(val_file)
            val_count += len(val_buffer)
    if train_count < 1000 or val_count < 100:
        raise RuntimeError("Corpus too small; collect more documents before pretraining.")
    cache_manifest.write_text(json.dumps(expected, indent=2))
    return train_count, val_count
