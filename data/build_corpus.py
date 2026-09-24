"""Build a clean CrazeAI training corpus from local JSONL/JSON/TXT files.

This intentionally does not download or copy another model's weights. It prepares
instruction/chat data for training the independent CrazeAI architecture.
"""
from __future__ import annotations
import json, hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "craze_train.jsonl"

SYSTEM = "You are CrazeAI, a precise, helpful assistant. Do not invent facts. Explain clearly and show working code when asked."

def norm(s: str) -> str:
    s = re.sub(r"\\s+", " ", str(s)).strip()
    return s

def valid(messages):
    if not isinstance(messages, list) or len(messages) < 2: return False
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return roles.count("user") >= 1 and roles.count("assistant") >= 1

def convert(obj):
    if isinstance(obj, dict) and valid(obj.get("messages", [])):
        msgs = obj["messages"]
    elif isinstance(obj, dict) and "instruction" in obj and ("output" in obj or "response" in obj):
        msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":norm(obj["instruction"])}, {"role":"assistant","content":norm(obj.get("output", obj.get("response")))}]
    elif isinstance(obj, dict) and "question" in obj and "answer" in obj:
        msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":norm(obj["question"])}, {"role":"assistant","content":norm(obj["answer"])}]
    else: return None
    out=[]
    for m in msgs:
        role=m.get("role"); content=norm(m.get("content", ""))
        if role in {"system","user","assistant"} and content: out.append({"role":role,"content":content})
    return out if valid(out) else None

def main():
    RAW.mkdir(exist_ok=True)
    seen=set(); rows=[]
    for p in RAW.rglob("*"):
        if p.suffix.lower() not in {".jsonl",".json"}: continue
        try:
            if p.suffix.lower()==".jsonl": objs=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            else:
                x=json.loads(p.read_text(encoding="utf-8")); objs=x if isinstance(x,list) else [x]
            for obj in objs:
                m=convert(obj)
                if not m: continue
                key=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
                if key not in seen: seen.add(key); rows.append({"messages":m})
        except Exception as e: print(f"skip {p}: {e}")
    with OUT.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"wrote {len(rows)} conversations to {OUT}")

if __name__ == "__main__": main()
