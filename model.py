import math
import torch
from torch import nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        a = a.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        a = F.softmax(a, dim=-1)
        a = self.drop(a)
        y = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
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

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrazeTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block_size = c["block_size"]
        n = c["n_embd"]
        self.tok = nn.Embedding(c["vocab_size"], n)
        self.pos = nn.Embedding(self.block_size, n)
        self.drop = nn.Dropout(c["dropout"])
        self.blocks = nn.ModuleList([
            Block(n, c["n_head"], self.block_size, c["dropout"])
            for _ in range(c["n_layer"])
        ])
        self.ln = nn.LayerNorm(n)
        self.head = nn.Linear(n, c["vocab_size"], bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        print(f"CrazeAI parameters: {self.parameters_count():,}")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, ignore_index=-100):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss
