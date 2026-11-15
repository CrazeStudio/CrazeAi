from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class CrazeAIConfig:
    vocab_size: int = 16000
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size < 256 or self.max_seq_len < 8:
            raise ValueError("vocab_size must be >=256 and max_seq_len >=8")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class DecoderBlock(nn.Module):
    def __init__(self, config: CrazeAIConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attn(
            normalized, normalized, normalized, attn_mask=causal_mask,
            need_weights=False, is_causal=True,
        )
        x = x + self.dropout(attended)
        return x + self.mlp(self.norm2(x))


class CrazeAIModel(nn.Module):
    """Decoder-only causal language model initialized from random weights."""

    def __init__(self, config: CrazeAIConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds {self.config.max_seq_len}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.final_norm(x))

    @classmethod
    def from_pretrained_dir(cls, directory: str | Path, device: torch.device):
        directory = Path(directory)
        config = CrazeAIConfig(**json.loads((directory / "config.json").read_text()))
        model = cls(config)
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.to(device)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        torch.save(self.state_dict(), directory / "model.pt")
