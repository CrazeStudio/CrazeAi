"""Text generation helpers."""

from __future__ import annotations

import torch

from .model import CrazeGPT
from .tokenizer import CharTokenizer


def generate(model: CrazeGPT, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[:, -model.block_size :]
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)

    return tokenizer.decode(ids[0].tolist())
