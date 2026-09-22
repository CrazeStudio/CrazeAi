import argparse
import json

import torch

from model import CrazeTransformer
from tokenizer import ByteTokenizer


def load_model():
    checkpoint = torch.load("crazeai.pt", map_location="cpu", weights_only=False)
    model = CrazeTransformer(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def clean_answer(text):
    marker = "### Output:\n"
    if marker in text:
        text = text.split(marker, 1)[1]

    # Stop if the model starts generating another training section.
    for stop in ("### Instruction:", "### Input:", "### Output:"):
        if stop in text:
            text = text.split(stop, 1)[0]

    # Remove control bytes that commonly appear as garbage.
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.65)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--greedy", action="store_true")
    a = p.parse_args()

    model = load_model()
    tokenizer = ByteTokenizer()

    formatted = (
        "### Instruction:\n"
        + a.prompt.strip()
        + "\n### Input:\n\n### Output:\n"
    )

    x = torch.tensor(
        [tokenizer.encode(formatted, add_bos=True, add_eos=False)],
        dtype=torch.long,
    )

    generated = []

    with torch.no_grad():
        for _ in range(a.max_new_tokens):
            logits, _ = model(x[:, -model.block_size:])
            scores = logits[:, -1, :].float()

            # Reserved IDs must never be emitted as normal answer text.
            scores[:, tokenizer.bos] = float("-inf")

            if a.greedy:
                nxt = torch.argmax(scores, dim=-1, keepdim=True)
            else:
                scores = scores / max(a.temperature, 0.1)
                k = min(max(1, a.top_k), scores.size(-1))
                values, indices = torch.topk(scores, k)
                filtered = torch.full_like(scores, float("-inf"))
                filtered.scatter_(1, indices, values)
                probs = torch.softmax(filtered, dim=-1)
                nxt = torch.multinomial(probs, 1)

            token = int(nxt.item())

            if token == tokenizer.eos:
                break

            generated.append(token)
            x = torch.cat([x, nxt], dim=1)

            # Avoid obvious one-character loops such as "aaaaaaaa...".
            if len(generated) >= 12:
                recent = generated[-12:]
                if len(set(recent)) == 1:
                    break

            # Also stop when the same short pattern repeats excessively.
            if len(generated) >= 24:
                tail = generated[-24:]
                if tail[:8] == tail[8:16] == tail[16:24]:
                    break

    answer = tokenizer.decode(generated)
    answer = clean_answer(answer)

    if not answer:
        answer = "I don't have enough trained information to answer that yet."

    print(answer)


if __name__ == "__main__":
    main()
