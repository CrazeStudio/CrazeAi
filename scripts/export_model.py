from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CrazeAI checkpoint as a standalone Transformers model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/crazeai-500m"))
    parser.add_argument("--output", type=Path, default=Path("outputs/crazeai-500m-export"))
    parser.add_argument("--merge-adapter", action="store_true", help="Merge a PEFT LoRA adapter into its base model.")
    args = parser.parse_args()

    if not args.model.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model}")
    adapter_config = args.model / "adapter_config.json"
    if adapter_config.is_file():
        if not args.merge_adapter:
            raise RuntimeError("This checkpoint is a LoRA adapter. Pass --merge-adapter to create a standalone model.")
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(args.model)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload(safe_merge=True)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True, trust_remote_code=False)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output)

    report_path = args.model / "training_report.json"
    export_report = {
        "source_checkpoint": str(args.model),
        "export_path": str(args.output),
        "adapter_merged": adapter_config.is_file(),
        "format": "Hugging Face Transformers safetensors",
        "base_model": getattr(model.config, "_name_or_path", None),
    }
    if report_path.is_file():
        shutil.copy2(report_path, args.output / "training_report.json")
    (args.output / "export_report.json").write_text(json.dumps(export_report, indent=2) + "\n", encoding="utf-8")

    # Validate that the exported directory is loadable before the workflow uploads it.
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.output,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if not any(True for _ in reloaded.parameters()):
        raise RuntimeError("Export validation failed: no model parameters found.")
    del reloaded
    print(json.dumps(export_report, indent=2))


if __name__ == "__main__":
    main()
