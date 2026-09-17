"""Small decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        y, _ = self.attn(x, x, x, attn_mask=self.mask[:t, :t], need_weights=False)
        return y


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CrazeGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 128, n_embd: int = 128, n_head: int = 4, n_layer: int = 2, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*(Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.block_size}")
        positions = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
