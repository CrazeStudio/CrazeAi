# CrazeAI 135M training pipeline

This pipeline builds a real CrazeAI model from an open **135M pretrained base**, then performs supervised fine-tuning on a compact, high-quality instruction mixture and exports a GGUF Q4 model for llama.cpp/Android.

## What is actually CrazeAI here?

The starting checkpoint is `HuggingFaceTB/SmolLM2-135M`, an Apache-2.0, 135M-parameter Llama-architecture base model. CrazeAI is the resulting fine-tuned checkpoint plus its own system behavior, dataset mixture, evaluation suite and tool/runtime layer. It does **not** copy Qwen weights.

The base model is deliberately kept in the 65M–150M target so the final Q4 GGUF stays around the ~100 MB class. SmolLM2-135M itself was pretrained on a large corpus before release; doing that pretraining from scratch for CrazeAI would require substantial GPU compute and would not be reproducible on a normal GitHub-hosted CPU runner.

## Pipeline

1. Download the 135M base checkpoint.
2. Prepare `HuggingFaceTB/smol-smoltalk` (the compact SFT dataset intended for sub-1B models) plus Craze-specific seed examples.
3. Supervised fine-tune with packed sequences and assistant-only loss.
4. Run held-out perplexity and deterministic capability probes.
5. Convert the resulting Transformers checkpoint to GGUF F16.
6. Quantize to `Q4_K_M` with llama.cpp.
7. Publish the GGUF and evaluation report as workflow artifacts.

The dataset is downloaded by the training runner, not committed to this repository.

## Training hardware

A 135M model is small for inference but SFT still needs a GPU to train in a reasonable time. The full workflow therefore targets a **self-hosted Linux CUDA runner** labelled `gpu`. A normal GitHub-hosted runner is only used for code validation/smoke tests; it should not silently spend hours attempting full training on CPU.

Required runner software: Python 3.11+, CUDA-capable PyTorch, Git, CMake, and enough disk for the base model, dataset cache and GGUF conversion.

## Output

The workflow produces:

- `crazeai-135m-f16/` — fine-tuned Transformers checkpoint
- `crazeai-135m-q4_k_m.gguf` — mobile-friendly quantized model
- `evaluation.json` — held-out loss and capability probe results
- `crazeai-system-prompt.txt` — runtime system behavior

## Important

A small model cannot become a frontier model simply by changing its prompt. The goal here is a **strong, honest on-device assistant** at roughly 135M parameters. Current/fresh information belongs in the web-search tool layer rather than being memorized into the small local model.
