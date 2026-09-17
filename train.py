import argparse
import math
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader

from model import CrazeGPT, GPTConfig
from tokenizer import CharTokenizer


class TextDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, i):
        return self.ids[i:i+self.block_size], self.ids[i+1:i+self.block_size+1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data.txt')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--n-embd', type=int, default=128)
    p.add_argument('--n-head', type=int, default=4)
    p.add_argument('--n-layer', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--out', default='model.pt')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not os.path.exists(args.data):
        raise FileNotFoundError(f'Training data not found: {args.data}')
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = CharTokenizer().train(text)
    tokenizer.save('vocab.json')
    ids = tokenizer.encode(text)
    if len(ids) <= args.block_size + 1:
        raise ValueError('Training data is too small for the selected block size')

    split = max(args.block_size + 1, int(len(ids) * 0.9))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(val_ids) <= args.block_size:
        val_ids = ids[-(args.block_size + 1):]

    dataset = TextDataset(train_ids, args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(tokenizer.vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout)
    model = CrazeGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = math.inf
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vx = torch.tensor(val_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                vy = torch.tensor(val_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                _, val_loss = model(vx[:, -args.block_size:], vy[:, -args.block_size:])
            model.train()
            print(f'step {step}/{args.steps} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}')
            if val_loss.item() < best:
                best = val_loss.item()
                torch.save({'config': vars(config), 'model': model.state_dict()}, args.out)

    if not os.path.exists(args.out):
        torch.save({'config': vars(config), 'model': model.state_dict()}, args.out)
    print(f'Saved {args.out} on {device}')


if __name__ == '__main__':
    main()
