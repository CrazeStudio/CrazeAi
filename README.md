# CrazeAI

CrazeAI is a CrazeStudio assistant built with **PyTorch**. This repository contains the reproducible dataset builder, full-parameter supervised fine-tuning pipeline, evaluation, CLI, and an OpenAI-compatible API.

## Model decision

- Target class: **500M parameters**.
- Checkpoint: \`Qwen/Qwen2.5-0.5B-Instruct\` (**0.49B actual parameters**, approximately 500M).
- Framework: PyTorch through Hugging Face Transformers.
- Training: full-parameter supervised fine-tuning by default; no random reinitialization of a 500M model.
- Dataset: exactly **50,000 unique examples**, with a deterministic 98%/2% train/validation split.
- Context used for training: 2,048 tokens by default. The base checkpoint supports a longer context, but training at 2,048 is a deliberate memory/cost choice.

CrazeAI is a new project and fine-tuned model identity, not a foundation model pretrained from zero. The model card and runtime must disclose its upstream checkpoint when asked about lineage.

## Dataset handling

The pipeline first looks for \`data/database.jsonl\`. This is the supported location for an existing local database. Accepted records may contain a \`messages\` array (roles: \`system\`, \`user\`, \`assistant\`) or instruction/input/output fields.

**The repository snapshot recovered from Git history did not contain a 50K local database**; it contained only 22 legacy seed examples. Therefore, when \`data/database.jsonl\` is absent, the data builder uses the Apache-2.0 SmolTalk subsets \`smol-magpie-ultra\`, \`smol-rewrite\`, and \`smol-summarize\`, then adds the small CrazeAI seed set. Dataset source, selected counts, split sizes, and file hashes are written to \`data/processed/manifest.json\`. Upstream data is downloaded at run time and is not committed to this repository.

The default mix is 31,992 Smol-Magpie-Ultra + 8,998 Smol-rewrite + 8,998 Smol-summarize + 12 CrazeAI seed records = 50,000 records. If the source changes or cannot provide the required unique records, preparation stops instead of silently training on fewer examples.

## Quick start

Use Python 3.11 or newer. Install a PyTorch build suitable for your machine, then install project dependencies:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

On Windows, activate the environment with \`.venv\\Scripts\\activate\`.

### 1. Prepare exactly 50K examples

    python scripts/prepare_dataset.py --count 50000

To use your own existing database, put it at \`data/database.jsonl\`. The builder automatically prefers that file. You can also select the source explicitly:

    python scripts/prepare_dataset.py --source local --input data/database.jsonl --count 50000

The local file must contain at least enough unique valid records to fill the requested count after the CrazeAI seed records are included.

### 2. Train on a CUDA GPU

    python scripts/train.py --train data/processed/train.jsonl --validation data/processed/validation.jsonl --output outputs/crazeai-500m --epochs 2 --max-length 2048 --batch-size 1 --gradient-accumulation 16

Full fine-tuning is memory-intensive. Use a CUDA GPU with **at least 16 GB VRAM recommended** for the default settings; lower-memory cards may require a smaller sequence length or batch/accumulation changes. CPU training is disabled by default because a 500M full fine-tune on CPU is impractically slow. \`--allow-cpu\` is available only for deliberate experiments.

### 3. Evaluate

    python scripts/evaluate.py --model outputs/crazeai-500m --validation data/processed/validation.jsonl --output outputs/crazeai-500m/evaluation.json

Evaluation reports token-weighted validation loss, perplexity, and deterministic chat probes. A successful training run is not a guarantee of a particular benchmark score; inspect the report and sample outputs before release.

### 4. Chat locally

    python scripts/chat.py --model outputs/crazeai-500m

### 5. Start the API

    export CRAZEAI_MODEL=outputs/crazeai-500m
    export CRAZEAI_API_KEY=replace-with-a-long-random-secret
    uvicorn crazeai.api:app --host 127.0.0.1 --port 8000

Health check: \`GET /health\`

Chat endpoint: \`POST /v1/chat/completions\`

Example request:

    curl http://127.0.0.1:8000/v1/chat/completions \\
      -H "Authorization: Bearer $CRAZEAI_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"Hello, who are you?"}],"max_tokens":128}'

For remote deployment, bind to the platform-provided port, set a strong \`CRAZEAI_API_KEY\`, and use a machine with enough RAM/VRAM for the model. Do not expose an unauthenticated API to the public internet.

## GitHub Actions

- \`.github/workflows/ci.yml\` runs syntax checks and unit tests on a standard CPU runner.
- \`.github/workflows/train-pytorch.yml\` is manually dispatched on a self-hosted Linux CUDA runner labelled \`gpu\`. The runner must have a working NVIDIA driver and CUDA-compatible PyTorch environment. The workflow uploads the trained checkpoint and evaluation report as artifacts; it does not commit model weights to Git.

## Repository layout

    crazeai/                  Runtime, prompt and data utilities
    scripts/                  Prepare, train, evaluate and chat
    data/crazeai_seed.jsonl   Small project identity/quality seed set
    data/database.jsonl       Optional user-provided existing database
    data/processed/           Generated train/validation data and manifest
    tests/                    Offline unit tests
    .github/workflows/        CI and GPU training workflows

## Important limitations

- 50K instruction examples are for **fine-tuning**, not for training a capable 500M foundation model from random initialization.
- The 0.49B checkpoint is pretrained by its upstream authors. CrazeAI's contribution is the fine-tuning data, training recipe, evaluation, system behavior and runtime integration.
- The 50K dataset is not currently committed as a local file. The exact local database path and public fallback are documented above.
- No trained CrazeAI weights are included until a GPU training run completes and passes evaluation.
- Model quality, speed and memory use depend on the training hardware, the actual dataset quality and the resulting checkpoint. No zero-error claim is possible.

## Upstream references

- Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Dataset: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org/
