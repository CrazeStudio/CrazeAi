# CrazeAI — scratch-trained language model

CrazeAI is a decoder-only language model built by CrazeStudio. Its model architecture, tokenizer and weights are created by this repository. **The model is initialized with random weights**: it does not load Qwen, Llama, Gemma or another pretrained checkpoint.

This repository now focuses on foundation-model pretraining. The previous Qwen/LoRA supervised-fine-tuning, adapter merge and GGUF export pipeline has been retired.

## Architecture

- Decoder-only causal Transformer implemented in native PyTorch.
- Learned token and positional embeddings, causal multi-head self-attention, MLP blocks and tied input/output embeddings.
- BPE tokenizer trained from the collected corpus; no pretrained tokenizer is downloaded.
- Default research configuration: 8 layers, width 256, 8 attention heads, 512-token context and a tokenizer vocabulary capped at 16,000.
- Default size is a small research model, not a 500M model. Change dimensions only after checking GPU memory and throughput.
- The model is a base language model. Next-token pretraining alone does not make it a reliable instruction-following chatbot; supervised instruction training and safety evaluation are later optional stages.

## Training stages

Use Python 3.11+. Install a PyTorch build appropriate for your CUDA device first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1. Collect online text

The default collector streams the FineWeb-Edu `sample-10BT` configuration from Hugging Face Datasets. It normalizes whitespace, removes exact duplicates and filters very short documents. It writes a corpus file and provenance manifest.

```bash
python scripts/collect_corpus.py --limit 100000
```

Increase `--limit` for larger corpora. The collector streams data; it does not download the entire dataset at once. Review the upstream dataset card, source provenance, license and applicable terms before training or publishing weights. Publicly accessible text is not automatically public-domain or free of copyright restrictions. Do not ingest private, personal or access-controlled material.

### 2. Train CrazeAI's tokenizer

```bash
python scripts/train_tokenizer.py --vocab-size 16000
```

The tokenizer is saved to `outputs/crazeai-base/tokenizer.json`. Its actual vocabulary size is read by the pretraining script so the embedding matrix matches it exactly.

### 3. Pretrain from random initialization

```bash
python scripts/pretrain.py --steps 10000 --batch-size 8 --block-size 512
```

The training loop uses next-token prediction, AdamW, gradient clipping, mixed precision on CUDA, periodic validation loss/perplexity, and model/optimizer checkpoints. To continue from the last saved state in the same output directory:

```bash
python scripts/pretrain.py --steps 20000 --batch-size 8 --block-size 512 --resume
```

Here `--steps` is the target total step, not additional steps. The default output directory is `outputs/crazeai-base`.

**Important:** 10,000 steps and 100,000 documents are starter settings for proving the pipeline, not enough to claim a capable general-purpose foundation model. A useful model requires substantially more high-quality tokens, longer training, careful validation and benchmark testing. Measure actual tokens/second on the target GPU and scale the run accordingly. Training from scratch generally costs much more compute than fine-tuning.

### 4. Chat with the base model

After a successful pretraining run:

```bash
python scripts/chat.py --model outputs/crazeai-base
```

This is a raw base model. Its output may continue text rather than follow instructions. That is expected before instruction tuning.

## Kaggle / GPU workflow

The GitHub Actions workflow **CrazeAI Scratch Pretraining** is manually dispatched and requires a persistent self-hosted Linux CUDA runner labelled `gpu`. It collects the corpus, trains the tokenizer, and runs scratch pretraining. GitHub-hosted standard runners do not provide a guaranteed GPU. If using Kaggle, run the same three commands in a notebook with a GPU accelerator and save `outputs/crazeai-base` plus `data/corpus/raw.manifest.json` to persistent storage between sessions.

## Checkpoints and reproducibility

The model checkpoint contains `config.json` and `model.pt`; the tokenizer is stored alongside it as `tokenizer.json`. Training state is saved separately as `training_state.pt` and includes optimizer state and the current step. Keep the corpus manifest with each model release. The corpus itself and model weights are generated artifacts and should not be committed to Git.

## Current limitations

- This is an initial, small scratch-trained architecture, not a production-grade 500M/1B assistant.
- The default model is not instruction-tuned and has no built-in live web access or tools.
- Corpus quality, license compliance, deduplication, language balance and contamination checks materially affect model quality.
- The current trainer is single-GPU; multi-GPU distributed training is a future optimization.
- No trained weights are included in the repository until a training run completes and is evaluated.
- The API can serve a local scratch checkpoint after it has been produced; it does not download a fallback pretrained model.

## API

Set the checkpoint directory and an API key, then run:

```bash
export CRAZEAI_MODEL=outputs/crazeai-base
export CRAZEAI_API_KEY='use-a-long-random-secret'
export APP_ENV=production
uvicorn crazeai.api:app --host 127.0.0.1 --port 8000
```

- Health: `GET /health`
- Chat: `POST /v1/chat/completions`

Do not expose a public endpoint without authentication and rate limiting.

## Repository map

```text
crazeai/model.py             Native decoder-only Transformer
crazeai/runtime.py           Local tokenizer + model inference
scripts/collect_corpus.py    Online corpus streaming and filtering
scripts/train_tokenizer.py   Train CrazeAI BPE tokenizer
scripts/pretrain.py          Random-initialization next-token training
scripts/chat.py              Local generation CLI
tests/                       Offline architecture tests
.github/workflows/pretrain.yml  Manual GPU pretraining workflow
```
