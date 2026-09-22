import argparse
import json

import torch

from model import CrazeTransformer
from tokenizer import ByteTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    a = p.parse_args()

    checkpoint = torch.load("crazeai.pt", map_location="cpu", weights_only=False)
    model = CrazeTransformer(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    formatted = (
        "### Instruction:\n"
        + a.prompt
        + "\n### Input:\n\n### Output:\n"
    )
    x = torch.tensor(
        [tokenizer.encode(formatted, add_bos=True, add_eos=False)],
        dtype=torch.long,
    )

    with torch.no_grad():
        for _ in range(a.max_new_tokens):
            logits, _ = model(x[:, -model.block_size:])
            scores = logits[:, -1, :] / max(a.temperature, 0.05)

            # Never generate the BOS token inside an answer.
            scores[:, tokenizer.bos] = float("-inf")

            k = min(max(1, a.top_k), scores.size(-1))
            values, indices = torch.topk(scores, k)
            filtered = torch.full_like(scores, float("-inf"))
            filtered.scatter_(1, indices, values)
            probs = torch.softmax(filtered, dim=-1)
            nxt = torch.multinomial(probs, 1)

            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == tokenizer.eos:
                break

    answer = tokenizer.decode(x[0].tolist())
    marker = "### Output:\n"
    if marker in answer:
        answer = answer.split(marker, 1)[1]
    print(answer.strip())


if __name__ == "__main__":
    main()
