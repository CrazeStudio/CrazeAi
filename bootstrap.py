"""Create a small CPU-friendly CrazeAI checkpoint when one is not present."""
import os
import subprocess
import sys

MODEL = "model.pt"
VOCAB = "vocab.json"

if os.path.exists(MODEL) and os.path.exists(VOCAB):
    print("CrazeAI checkpoint already exists; skipping bootstrap training.")
    raise SystemExit(0)

print("No checkpoint found. Training a tiny starter CrazeAI model...")
subprocess.run([
    sys.executable, "train.py",
    "--data", "data.txt",
    "--steps", os.getenv("BOOTSTRAP_STEPS", "300"),
    "--batch-size", "8",
    "--block-size", "64",
    "--n-embd", "96",
    "--n-head", "4",
    "--n-layer", "2",
    "--dropout", "0.0",
    "--lr", "0.0005",
    "--out", MODEL,
], check=True)

if not (os.path.exists(MODEL) and os.path.exists(VOCAB)):
    raise RuntimeError("Bootstrap training finished without creating model.pt and vocab.json")
print("CrazeAI starter checkpoint created successfully.")
