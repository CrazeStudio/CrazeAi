from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "ckpt.pt"
TOKENIZER = ROOT / "models" / "tokenizer.json"
BASE_URL = "https://huggingface.co/brettleehari/microgpt/resolve/main/"
MODEL.parent.mkdir(parents=True, exist_ok=True)
for path, name in ((MODEL, "ckpt.pt"), (TOKENIZER, "tokenizer.json")):
    if not path.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(BASE_URL + name, path)
    print(f"Ready: {path}")
