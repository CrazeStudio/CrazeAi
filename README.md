# CrazeAI

CrazeAI is a small PyTorch AI service built around the MIT-licensed microGPT checkpoint, with an analyzer/controller layer and optional web search.

## Architecture

User -> Analyzer -> Controller -> Search (when needed) -> microGPT -> Response

The bundled checkpoint is a 1.35M-parameter GPT-style base model trained on TinyStories. It is not instruction-tuned; the analyzer, controller, and search layer provide orchestration and current-information retrieval.

## Render

Build: `pip install -r requirements.txt && python setup_model.py`

Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

The build downloads the matching public tokenizer if it is not already present.

## API

`GET /health`

`POST /generate`

```json
{"prompt":"What is the latest Minecraft version?","search":true}
```
