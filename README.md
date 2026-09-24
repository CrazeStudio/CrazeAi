# CrazeAI

CrazeAI is a compact AI assistant project focused on **useful intelligence at small model size** and eventual local Android inference.

## Repository structure

```text
CrazeAi/
├── training_v2/          # Current model training pipeline
│   ├── prepare_sft.py    # Dataset preparation + CrazeAI behavior/identity
│   ├── train_sft.py      # Supervised fine-tuning
│   ├── evaluate.py       # Model evaluation
│   ├── crazeai_identity.jsonl
│   └── crazeai-system-prompt.txt
├── android/              # Android + native llama.cpp runtime
├── backend/              # Optional server/tool layer
├── .github/workflows/    # Automated training/quantization workflow
├── data/                 # Reserved for active project data
├── legacy/               # Old experimental model/training code
└── README.md
```

The old root-level experimental Python files have been moved into `legacy/` so they cannot be confused with the current pipeline.

## Model direction

`HuggingFaceTB/SmolLM2-135M` → high-quality SFT mixture → CrazeAI behavior/identity tuning → evaluation → GGUF → Q4_K_M → Android/llama.cpp.

The base model is used as a starting point; CrazeAI is the resulting fine-tuned model with Craze-specific training, behavior and evaluation. CrazeAI must not claim to be Qwen, SmolLM, ChatGPT, or another assistant.

## What CrazeAI is being taught

The training mixture emphasizes:

- instruction following
- coding and debugging
- explanations
- reasoning and problem solving
- concise normal conversation
- accuracy and uncertainty handling
- current-information/tool awareness
- CrazeAI's own identity and model lineage

For example, CrazeAI is explicitly trained to answer **"Who are you?"** with its own identity rather than the identity of its base model.

## Training

The main workflow is:

```text
prepare SFT data
      ↓
135M supervised fine-tuning
      ↓
evaluation
      ↓
GGUF export
      ↓
Q4_K_M quantization
      ↓
Android llama.cpp runtime
```

Use `.github/workflows/train-crazeai-135m.yml` for the automated pipeline. The workflow should run on an appropriate GPU runner for practical full training; CPU fallback exists for compatibility but is substantially slower.

## Runtime/tool layer

The small local model should not be forced to memorize everything. The Android/tool layer can route current or external-information questions to web search, provide evidence to CrazeAI, and let the model synthesize the final answer.

Simple prompts such as `hello` should remain fast and should not trigger unnecessary deep reasoning or web search.
