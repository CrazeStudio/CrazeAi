from __future__ import annotations

import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import CrazeAIModel

DEFAULT_LOCAL_MODEL = Path("outputs/crazeai-base")


class CrazeAIRuntime:
    def __init__(self, model_name: str | None = None):
        self.model_name = str(model_name or os.getenv("CRAZEAI_MODEL") or DEFAULT_LOCAL_MODEL)
        directory = Path(self.model_name).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"CrazeAI checkpoint not found: {directory}. Run corpus collection, "
                "tokenizer training and scratch pretraining first."
            )
        tokenizer_path = directory / "tokenizer.json"
        if not tokenizer_path.is_file() or not (directory / "model.pt").is_file():
            raise FileNotFoundError(f"{directory} must contain tokenizer.json and model.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.model = CrazeAIModel.from_pretrained_dir(directory, self.device)
        if self.tokenizer.get_vocab_size() != self.model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match the model config")
        self.model.eval()
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=128, temperature=0.8, top_p=0.9):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if not 1 <= max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        if not 0.0 <= temperature <= 2.0 or not 0.05 <= top_p <= 1.0:
            raise ValueError("Invalid temperature or top_p")
        text = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages if isinstance(m, dict)
        ) + "\nASSISTANT:"
        ids = self.tokenizer.encode(text).ids
        if self.bos_id is not None:
            ids.insert(0, self.bos_id)
        if not ids:
            raise ValueError("Prompt produced no tokens")
        prompt_count = len(ids)
        generated = []
        for _ in range(max_new_tokens):
            context = ids[-self.model.config.max_seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=self.device)
            logits = self.model(x)[0, -1].float()
            if temperature == 0:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                sorted_probs[cumulative - sorted_probs > top_p] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_id = int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())
            if next_id == self.eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
        return self.tokenizer.decode(generated).strip(), prompt_count, len(generated)
