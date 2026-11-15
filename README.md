# CrazeAI

CrazeAI is a CrazeStudio assistant built with **PyTorch**. This repository contains the reproducible dataset builder, full-parameter supervised fine-tuning pipeline, evaluation, CLI, and an OpenAI-compatible API.

## Model decision

- Target class: **500M parameters**.
- Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct` (**0.49B actual parameters**, approximately 500M).
- Framework: PyTorch through Hugging Face Transformers.
- Training: full-parameter supervised fine-tuning by default; no random reinitialization of a 500M model.
- Dataset: exactly **50,000 unique examples**, with a deterministic 98%/2% train/validation split.
- Context used for training: 2,048 tokens by default. The base checkpoint supports a longer context, but training at 2,048 is a deliberate memory/cost choice.

CrazeAI is a new project and fine-tuned model identity, not a foundation model pretrained from zero. The model card and runtime must disclose its upstream checkpoint when asked about lineage.

## Dataset handling

The pipeline first looks for `data/database.jsonl`. This is the supported location for an existing local database. Accepted records may contain a `messages` array (roles: `system`, `user`, `assistant`) or instruction/input/output fields.

**The repository snapshot recovered from Git history did not contain a 50K local database**; it contained only 22 legacy seed examples. Therefore, when `data/database.jsonl` is absent, the data builder uses the Apache-2.0 SmolTalk subsets `smol-magpie-ultra`, `smol-rewrite`, and `smol-summarize`, then adds the small CrazeAI seed set. Dataset source, selected counts, split sizes, and file hashes are written to `data/processed/manifest.json`. Upstream data is downloaded at run time and is not committed to this repository.

The default mix is 31,992 Smol-Magpie-Ultra + 8,998 Smol-rewrite + 8,998 Smol-summarize + 12 CrazeAI seed records = 50,000 records. If the source changes or cannot provide the required unique records, preparation stops instead of silently training on fewer examples.

## Quick start

Use Python 3.11 or newer. Install a PyTorch build suitable for your machine, then install project dependencies:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

On Windows, activate the environment with `.venv\\Scripts\\activate`.

### 1. Prepare exactly 50K examples

    python scripts/prepare_dataset.py --count 50000

To use your own existing database, put it at `data/database.jsonl`. The builder automatically prefers that file. You can also select the source explicitly:

    python scripts/prepare_dataset.py --source local --input data/database.jsonl --count 50000

The local file must contain at least enough unique valid records to fill the requested count after the CrazeAI seed records are included.

### 2. Train on one CUDA GPU

    python -m scripts.train --train data/processed/train.jsonl --validation data/processed/validation.jsonl --output outputs/crazeai-500m --epochs 1 --max-length 512 --batch-size 2 --gradient-accumulation 4

### 3. Train on two CUDA GPUs (Kaggle T4 x2)

Select **GPU T4 x2** in Kaggle Notebook Settings → Accelerator, restart the session, and verify that PyTorch sees two devices:

    import torch
    print(torch.cuda.device_count())
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

The device count must be `2`. Launch one process per GPU with DDP:

    torchrun --standalone --nproc_per_node=2 -m scripts.train --train data/processed/train.jsonl --validation data/processed/validation.jsonl --output /kaggle/working/crazeai-500m-ddp --epochs 1 --max-steps 1200 --max-length 512 --batch-size 2 --gradient-accumulation 4

The two-GPU command uses rank-zero-only preprocessing to avoid concurrent writes, NCCL for GPU communication, and one Trainer process per GPU. It defaults to LoRA adapter fine-tuning (only small adapter matrices are trainable), a 512-token context, per-device batch 2, accumulation 4, and a 1,200-step cap. The effective batch size is 16 across two GPUs. The step cap is a time-budget guard, not a promise that every Kaggle T4 x2 session will finish within 12 hours; check the first 20–30 steps and estimate from measured throughput. LoRA adapters are saved with their base-model reference and are loaded automatically by the runtime. Use `--full-finetune` to train every parameter, but that is substantially slower and is not recommended for a 12-hour T4 budget.

The shorter 512-token context and LoRA mode are deliberate speed choices. Increase context only after measuring throughput. CPU training is disabled by default because full fine-tuning a 500M model on CPU is impractically slow. `--allow-cpu` is available only for deliberate experiments.

### 4. TPU note

The current training script targets CUDA and NCCL. It does **not** support TPU execution. TPU training needs a separate PyTorch TPU backend and distributed launch path; do not run this CUDA script unchanged on a TPU.

### 5. Evaluate

    python scripts/evaluate.py --model outputs/crazeai-500m --validation data/processed/validation.jsonl --output outputs/crazeai-500m/evaluation.json

Evaluation reports token-weighted validation loss, perplexity, and deterministic chat probes. A successful training run is not a guarantee of a particular benchmark score; inspect the report and sample outputs before release.

### 6. Chat locally

The runtime accepts either a merged Transformers model directory or a LoRA adapter directory (the adapter is loaded together with its recorded base model):

    python scripts/chat.py --model outputs/crazeai-500m

For a Kaggle LoRA run saved to `/kaggle/working/crazeai-test`, test the adapter directly:

    python -m scripts.chat --model /kaggle/working/crazeai-test

If you want a standalone merged export first:

    python -m scripts.export_model --model /kaggle/working/crazeai-test --output /kaggle/working/crazeai-test-export --merge-adapter
    python -m scripts.chat --model /kaggle/working/crazeai-test-export

If a supplied local path is missing or is not a model directory, the runtime now raises a clear `FileNotFoundError` listing nearby loadable checkpoints. When a requested path ends in `-export` but has not been created yet, CrazeAI checks for the matching training directory and loads its adapter directly if available, with a warning.

### 7. Export the CrazeAI Test release to GGUF

The GGUF exporter downloads the LoRA adapter from the GitHub release, merges it into `Qwen/Qwen2.5-0.5B-Instruct`, converts the merged model with llama.cpp, and quantizes it. It does not require a Kaggle training session; it only needs internet access and enough RAM and disk space to download and convert the base model.

Install the project dependencies first, then run:

    python scripts/export_gguf.py

The default output is `outputs/gguf/CrazeAI-Test-Q4_K_M.gguf`. To export an unquantized FP16 GGUF instead:

    python scripts/export_gguf.py --quantization F16

Other supported quantization choices are `Q5_K_M` and `Q8_0`. The script clones llama.cpp into `tools/llama.cpp` and builds its CPU quantizer when quantization is enabled. Use `--keep-intermediates` to retain the merged Transformers model and temporary FP16 GGUF. You can override the release asset with `--adapter-url` and the destination with `--output-dir`.

The exported GGUF is a standalone merged model; it no longer needs the LoRA adapter at inference time. Quantization reduces file size and can slightly affect output quality.



### 8. One-click GGUF export and release upload (recommended)

You do not need Kaggle for conversion. Open **Actions → CrazeAI GGUF Export → Run workflow**, keep release tag `v0.1.0-test`, choose `Q4_K_M`, and run it. The workflow uses a GitHub-hosted CPU runner, downloads the LoRA ZIP from the release, merges it into the Qwen base model, converts and quantizes it, then uploads the resulting `.gguf` and SHA-256 checksum to the same release. It requires the repository Actions workflow to have permission to write contents/releases.

You can choose `Q5_K_M`, `Q8_0`, or `F16` in the workflow input. The output asset is named `CrazeAI-Test-<QUANTIZATION>.gguf`.

For manual local/Kaggle export, run `python scripts/export_gguf.py --quantization Q4_K_M`. To upload directly from that environment, set `GH_TOKEN` to a token with Contents: read/write permission on this repository and add `--upload`.

### 9. Start the API

    export CRAZEAI_MODEL=outputs/crazeai-500m
    export CRAZEAI_API_KEY=replace-with-a-long-random-secret
    uvicorn crazeai.api:app --host 127.0.0.1 --port 8000

Health check: `GET /health`

Chat endpoint: `POST /v1/chat/completions`

Example request:

    curl http://127.0.0.1:8000/v1/chat/completions \\
      -H "Authorization: Bearer $CRAZEAI_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"Hello, who are you?"}],"max_tokens":128}'

For remote deployment, bind to the platform-provided port, set a strong `CRAZEAI_API_KEY`, and use a machine with enough RAM/VRAM for the model. Do not expose an unauthenticated API to the public internet.

## GitHub Actions

- `.github/workflows/ci.yml` runs syntax checks and unit tests on a standard CPU runner.
- `.github/workflows/train-pytorch.yml` is manually dispatched on a self-hosted Linux CUDA runner labelled `gpu`. The runner must have a working NVIDIA driver and CUDA-compatible PyTorch environment. The workflow uploads the trained checkpoint and evaluation report as artifacts; it does not commit model weights to Git.

## Repository layout

    crazeai/                  Runtime, prompt and data utilities
    scripts/                  Prepare, train, evaluate, chat and export
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


### Resumable training and standalone export

Run **Actions → CrazeAI Train, Resume and Export → Run workflow** on a configured self-hosted Linux CUDA runner labelled `gpu`. The workflow restores the latest retained training-state artifact, resumes from the newest `checkpoint-*` directory when enabled, evaluates the completed checkpoint, and exports a standalone Transformers model. LoRA adapters are merged into the base model for the standalone export; full fine-tuning checkpoints are exported directly.

The job timeout is 24 hours, but timeout settings do not extend the lifetime of a disconnected or externally terminated runner. The workflow uploads checkpoint and dataset state with `if: always()`, so a failed or cancelled run can be resumed from its retained artifact on a later run. Artifact retention is 30 days. The resumable state artifact contains the checkpoint, optimizer/scheduler state and processed dataset; the standalone export is a separate artifact named `crazeai-500m-standalone-export`.

A self-hosted GPU runner is required. GitHub Actions cannot create GPU capacity by setting a longer timeout. Protect the runner and its storage, and verify available disk space before a 50K-example run. A 50K SFT set fine-tunes the declared pretrained Qwen checkpoint; it does not train a foundation model from random initialization.
