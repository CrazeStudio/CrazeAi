import argparse
import os
import torch
from model import CrazeGPT, GPTConfig
from tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.txt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="model.pt")
    args = p.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training data not found: {args.data}")
    text = open(args.data, "r", encoding="utf-8").read()
    tokenizer = CharTokenizer().train(text)
    tokenizer.save("vocab.json")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(data) <= args.block_size + 1:
        raise ValueError("Training data is too small for the selected block size")

    config = GPTConfig(tokenizer.vocab_size, block_size=args.block_size, n_embd=128, n_head=4, n_layer=2, dropout=0.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrazeGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i + args.block_size] for i in starts]).to(device)
        y = torch.stack([data[i + 1:i + args.block_size + 1] for i in starts]).to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")

    torch.save({"config": vars(config), "model": model.state_dict()}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
