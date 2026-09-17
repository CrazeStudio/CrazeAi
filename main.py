import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
from model import CrazeGPT
from tokenizer import CharTokenizer
from generation import generate

app = FastAPI(title="CrazeAI", version="0.1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")
model = None
tokenizer = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=500)


def load_model():
    global model, tokenizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        return
    tokenizer = CharTokenizer.load(VOCAB_PATH)
    model = CrazeGPT.from_checkpoint(MODEL_PATH, map_location=DEVICE).to(DEVICE)
    model.eval()


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": DEVICE}


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/generate")
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model and provide model.pt and vocab.json.")
    text = generate(model, tokenizer, request.prompt, request.max_new_tokens, request.temperature, request.top_k, DEVICE)
    return {"response": text}
