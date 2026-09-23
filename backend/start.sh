#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p /tmp/crazeai
MODEL="/tmp/crazeai/Qwen3-1.7B-Q4_K_M.gguf"
URL="https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true"

if [ ! -s "$MODEL" ]; then
  echo "Downloading Qwen3-1.7B Q4_K_M..."
  curl -L --fail --retry 8 --retry-delay 3 --connect-timeout 20 --progress-bar "$URL" -o "$MODEL.part"
  mv "$MODEL.part" "$MODEL"
fi

if [ ! -x ./llama.cpp/build/bin/llama-server ]; then
  rm -rf llama.cpp
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
  cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
  cmake --build llama.cpp/build -j2 --target llama-server
fi

./llama.cpp/build/bin/llama-server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192 \
  --threads 2 \
  --parallel 1 \
  --jinja \
  --n-predict 768 \
  > /tmp/crazeai-llama.log 2>&1 &

PID=$!
for i in $(seq 1 600); do
  if ! kill -0 "$PID" 2>/dev/null; then
    cat /tmp/crazeai-llama.log
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 || { cat /tmp/crazeai-llama.log; exit 1; }
exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-10000}"
