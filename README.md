# CrazeAI

A small GPT-style language model built from scratch with Python and PyTorch, exposed through a FastAPI web API and a lightweight browser interface.

## Planned stack

- **PyTorch** — Transformer language model and training
- **Tokenizer** — simple trainable text tokenizer, designed to be replaceable with BPE/SentencePiece later
- **FastAPI** — inference API
- **HTML/CSS/JavaScript** — browser chat interface
- **Render** — deployment target

## Project goals

1. Train a compact causal language model from text data.
2. Tokenize prompts and generated text consistently.
3. Serve inference through `POST /generate`.
4. Provide a mobile-friendly chat page.
5. Keep the architecture small enough to experiment with on modest hardware.

## Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Status

🚧 Early development — model, tokenizer, training pipeline, API, and web UI are being built incrementally.
