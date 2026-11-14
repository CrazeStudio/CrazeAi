from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .runtime import CrazeAIRuntime

security = HTTPBearer(auto_error=False)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    expected = os.getenv("CRAZEAI_API_KEY")
    if not expected:
        if os.getenv("APP_ENV", "development").lower() == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is not configured",
            )
        return
    supplied = credentials.credentials if credentials else ""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer API key required")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("APP_ENV", "development").lower() == "production" and not os.getenv("CRAZEAI_API_KEY"):
        raise RuntimeError("CRAZEAI_API_KEY must be set when APP_ENV=production")
    app.state.runtime = CrazeAIRuntime()
    app.state.inference_lock = threading.Lock()
    yield
    app.state.runtime = None


app = FastAPI(
    title="CrazeAI API",
    version="1.0.0",
    description="PyTorch-powered CrazeAI 0.5B chat API.",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "name": "CrazeAI",
        "status": "ready" if getattr(app.state, "runtime", None) else "starting",
        "api": "/v1/chat/completions",
    }


@app.get("/health")
async def health():
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "status": "ok",
        "model": "crazeai-500m",
        "device": str(runtime.device),
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    messages = [item.model_dump() for item in request.messages]

    def infer():
        with app.state.inference_lock:
            return runtime.generate(
                messages,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

    try:
        answer, prompt_tokens, completion_tokens = await run_in_threadpool(infer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Model inference failed") from exc

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "crazeai-500m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "length" if completion_tokens >= request.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
