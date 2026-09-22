from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json, os, torch
from model import CrazeTransformer
from tokenizer import ByteTokenizer

app = FastAPI(title="CrazeAI API", version="1.0.0")
tokenizer = ByteTokenizer()
model = None

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.8, gt=0, le=2)
    top_k: int = Field(40, ge=1, le=256)

def load_model():
    global model
    if model is not None:
        return model
    if not os.path.exists("crazeai.pt"):
        raise HTTPException(status_code=503, detail="CrazeAI model checkpoint is not trained yet.")
    with open("config.json", encoding="utf-8") as f:
        config = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrazeTransformer(config).to(device)
    ckpt = torch.load("crazeai.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

@app.get("/")
def root():
    return {"name": "CrazeAI", "status": "online", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "checkpoint": os.path.exists("crazeai.pt")}

@app.post("/generate")
def generate(req: GenerateRequest):
    m = load_model()
    device = next(m.parameters()).device
    ids = tokenizer.encode(req.prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            logits, _ = m(x[:, -m.block_size:])
            logits = logits[:, -1, :] / max(req.temperature, 0.01)
            k = min(req.top_k, logits.size(-1))
            values, indices = torch.topk(logits, k)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            probs = torch.softmax(filtered, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == tokenizer.eos:
                break
    return {"text": tokenizer.decode(x[0].tolist()[len(ids):]), "model": "CrazeAI"}

@app.get("/model-info")
def model_info():
    with open("config.json", encoding="utf-8") as f:
        config = json.load(f)
    return {"name":"CrazeAI","architecture":"decoder-only causal Transformer","config":config,"checkpoint_available":os.path.exists("crazeai.pt")}
