from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import SYSTEM_PROMPT

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LOCAL_MODEL = Path("outputs/crazeai-500m")


def resolve_model_name(model_name: str | None = None) -> str:
    if model_name:
        return model_name
    configured = os.getenv("CRAZEAI_MODEL")
    if configured:
        return configured
    if DEFAULT_LOCAL_MODEL.is_dir():
        return str(DEFAULT_LOCAL_MODEL)
    return BASE_MODEL_ID


class CrazeAIRuntime:
    def __init__(self, model_name: str | None = None):
        self.model_name = resolve_model_name(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        adapter_config = Path(self.model_name) / "adapter_config.json"
        if adapter_config.is_file():
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(self.model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                trust_remote_code=False,
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                trust_remote_code=False,
            )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, int, int]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if not 1 <= max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.05 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.05 and 1")

        safe_messages: list[dict[str, str]] = []
        for message in messages[-64:]:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                continue
            if content.strip():
                safe_messages.append({"role": role, "content": content.strip()})

        if not safe_messages or not any(m["role"] == "user" for m in safe_messages):
            raise ValueError("At least one non-empty user message is required")
        if not any(m["role"] == "system" for m in safe_messages):
            safe_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        encoded = self.tokenizer.apply_chat_template(
            safe_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > 8192:
            raise ValueError("The conversation exceeds the 8,192-token API input limit")

        generation_args: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
            "use_cache": True,
        }
        if temperature == 0:
            generation_args["do_sample"] = False
        else:
            generation_args.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        output = self.model.generate(**encoded, **generation_args)
        generated_ids = output[0, prompt_tokens:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer, prompt_tokens, int(generated_ids.numel())
