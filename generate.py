import argparse
import json
import torch
from model import CrazeTransformer
from tokenizer import ByteTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    a = p.parse_args()

    ck = torch.load("crazeai.pt", map_location="cpu", weights_only=False)
    m = CrazeTransformer(ck["config"])
    m.load_state_dict(ck["model"])
    m.eval()
    t = ByteTokenizer()
    x = torch.tensor([t.encode(a.prompt, add_bos=True, add_eos=False)])

    with torch.no_grad():
        for _ in range(a.max_new_tokens):
            logits, _ = m(x[:, -m.block_size:])
            z = logits[:, -1, :] / max(a.temperature, 0.01)
            v, ix = torch.topk(z, min(a.top_k, z.size(-1)))
            z.fill_(float("-inf"))
            z.scatter_(1, ix, v)
            probs = torch.softmax(z, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == t.eos:
                break

    print(t.decode(x[0].tolist()))

if __name__ == "__main__":
    main()
