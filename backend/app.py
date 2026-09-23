import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CrazeAI", version="1.0")
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    max_tokens: int = 512
    temperature: float = 0.7

@app.get("/")
def root():
    return {"name": "CrazeAI", "model": "Qwen3-1.7B", "status": "online"}

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{LLAMA_URL}/health")
        return {"status": "ok" if r.is_success else "loading", "model": "ready" if r.is_success else "loading"}
    except Exception:
        return {"status": "loading", "model": "loading"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message is required")
    messages = [{"role": "system", "content": "You are CrazeAI, a helpful, concise and intelligent assistant. Answer naturally. For coding questions, give correct code and explain where it belongs. Do not repeat yourself."}]
    messages.extend(req.history[-12:])
    messages.append({"role": "user", "content": req.message})
    payload = {"messages": messages, "max_tokens": min(req.max_tokens, 1024), "temperature": req.temperature, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return {"response": text, "model": "Qwen3-1.7B"}
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")
