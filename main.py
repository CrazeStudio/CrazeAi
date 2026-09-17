import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from generation import generate
from model import CrazeGPT
from tokenizer import CharTokenizer

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="CrazeAI", version="0.2.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "model.pt")))
VOCAB_PATH = Path(os.getenv("VOCAB_PATH", str(BASE_DIR / "vocab.json")))
model = None
tokenizer = None
load_error = None

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=500)

def load_model():
    global model, tokenizer, load_error
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"Missing vocabulary file: {VOCAB_PATH}")
        tokenizer = CharTokenizer.load(str(VOCAB_PATH))
        model = CrazeGPT.from_checkpoint(str(MODEL_PATH), map_location=DEVICE).to(DEVICE)
        model.eval()
        load_error = None
        print(f"CrazeAI loaded: {MODEL_PATH}")
    except Exception as exc:
        model = None
        tokenizer = None
        load_error = str(exc)
        print(f"CrazeAI model load failed: {exc}")

@app.on_event("startup")
def startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE, "model_path": str(MODEL_PATH), "vocab_path": str(VOCAB_PATH), "error": load_error}

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.post("/generate")
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=f"CrazeAI model is unavailable: {load_error or 'unknown error'}")
    try:
        text = generate(model, tokenizer, request.prompt, request.max_new_tokens, request.temperature, request.top_k, DEVICE)
        return {"response": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
