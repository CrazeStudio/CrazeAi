# CrazeAI

A compact decoder-only Transformer for code generation and instruction learning.

## Files
- `model.py` — causal Transformer
- `tokenizer.py` — byte-level tokenizer
- `dataset.py` — JSONL-style instruction dataset loader
- `train.py` — training/checkpointing
- `generate.py` — sampling/inference
- `config.json` — model settings
- `data.json` — training data
- `requirements.txt` — dependencies

This is a new lightweight CrazeAI implementation. It does not contain Meta model weights.

### Train
```bash
pip install -r requirements.txt
python train.py
python generate.py --prompt "Write a Python function that reverses a string."
```
