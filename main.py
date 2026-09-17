from pathlib import Path
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from crazeai.model import load_model, generate
from crazeai.tokenizer import load_tokenizer
from crazeai.controller import run

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "ckpt.pt"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", ROOT / "models" / "tokenizer.json"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="CrazeAI", version="1.1.0")
model = None
tokenizer = None
load_error = None

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=64, ge=1, le=128)
    temperature: float = Field(default=0.7, gt=0, le=2)
    top_k: int = Field(default=30, ge=0, le=4096)
    search: bool = True

@app.on_event("startup")
def startup():
    global model, tokenizer, load_error
    try:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        model = load_model(MODEL_PATH, DEVICE)
        load_error = None
    except Exception as exc:
        load_error = str(exc); model = None; tokenizer = None

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":model is not None,"device":DEVICE,"model":"microGPT 1.35M","kv_cache":True,"streaming":True,"search":True,"error":load_error}

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

def response_text(req):
    if req.search:
        return run(req.prompt, model, tokenizer, lambda m,t,p,max_new_tokens: generate(m,t,p,max_new_tokens=max_new_tokens,temperature=req.temperature,top_k=req.top_k), req.max_new_tokens)
    return {"response":generate(model,tokenizer,req.prompt,req.max_new_tokens,req.temperature,req.top_k),"plan":{"action":"answer","needs_search":False},"sources":[]}

@app.post("/generate")
def generate_text(req: GenerateRequest):
    if model is None or tokenizer is None: raise HTTPException(503,f"CrazeAI unavailable: {load_error}")
    try: return response_text(req)
    except Exception as exc: raise HTTPException(500,f"Generation failed: {exc}") from exc
