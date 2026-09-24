# CrazeAI

CrazeAI is a lightweight Android AI project. The repository now contains two generations:

- **Legacy reference runtime:** the original small custom Transformer + `.craze` format in the root and `android/`.
- **CrazeAI 135M pipeline:** the recommended production direction in `training_v2/`, using a 135M Llama-compatible base, supervised fine-tuning, evaluation and GGUF Q4 export for llama.cpp.

## Recommended model pipeline

`HuggingFaceTB/SmolLM2-135M` → curated SFT mixture → CrazeAI behavior tuning → evaluation → GGUF F16 → `Q4_K_M` → Android/llama.cpp.

The 135M base is Apache-2.0 and is substantially smaller than the old Qwen3-1.7B experiment. CrazeAI does **not** claim the Qwen weights as its own. The resulting CrazeAI checkpoint is a fine-tuned 135M model with Craze-specific data, behavior and evaluation.

The compact `HuggingFaceTB/smol-smoltalk` dataset is used because it was specifically prepared for models below 1B parameters and contains roughly 460k training examples. It covers instruction following, rewriting, summarization, reasoning and other assistant behaviors. The training pipeline also adds Craze-specific coding, accuracy and assistant-behavior examples.

### Run the model pipeline

Use `.github/workflows/train-crazeai-135m.yml` from GitHub Actions. It performs syntax validation, downloads/prepares the SFT data, fine-tunes the 135M base, evaluates the result, converts it to GGUF and quantizes it to `Q4_K_M`.

Full training requires a CUDA GPU runner labelled `self-hosted`, `linux`, `x64`, `gpu`. A normal GitHub-hosted CPU runner is deliberately not used for the full training job because that would be impractically slow.

### Output

The workflow produces:

```text
crazeai-135m-q4_k_m.gguf
evaluation.json
crazeai-system-prompt.txt
```

The Q4 GGUF is the model artifact intended for the Android llama.cpp runtime. It is expected to be around the ~100 MB class rather than the old ~1 GB model.

## Web search and reasoning

Fresh information should be supplied by the application tool layer rather than memorized into a tiny local model. The Android assistant can route current/time-sensitive questions to web search, feed the retrieved evidence to CrazeAI, and ask the model to synthesize an answer with sources. Simple questions should stay local for speed.

## Legacy files

The original root files (`model.py`, `train.py`, `export.py`, etc.) are retained for compatibility and historical reference. The new GGUF pipeline lives under `training_v2/` and is the recommended path for new model releases.
