import json
import os

import torch
from torch.utils.data import DataLoader, random_split

from model import CrazeTransformer
from dataset import CrazeDataset


def save_checkpoint(path, model, optimizer, config, epoch, step):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def main():
    with open("config.json", encoding="utf-8") as f:
        c = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(c.get("seed", 42))

    dataset = CrazeDataset(c.get("dataset", "data.json"), c["block_size"])

    val_fraction = float(c.get("validation_fraction", 0.05))
    val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size

    if val_size:
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(c.get("seed", 42)),
        )
    else:
        train_ds, val_ds = dataset, None

    loader = DataLoader(
        train_ds,
        batch_size=c["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False)
        if val_ds is not None
        else None
    )

    model = CrazeTransformer(c).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=c["learning_rate"],
        weight_decay=c.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )

    start_epoch = 0
    resume = c.get("resume")
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"resumed from {resume} at epoch {start_epoch + 1}")

    use_amp = device.type == "cuda" and c.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, c["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / max(1, c.get("grad_accumulation", 1))

            scaler.scale(loss).backward()

            if (step + 1) % max(1, c.get("grad_accumulation", 1)) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * max(1, c.get("grad_accumulation", 1))

        avg = total_loss / max(1, len(loader))
        message = f"epoch {epoch + 1}/{c['epochs']} train_loss={avg:.4f}"

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        _, loss = model(x, y)
                    val_total += loss.item()
            message += f" val_loss={val_total / max(1, len(val_loader)):.4f}"

        print(message)
        save_checkpoint(c["checkpoint"], model, optimizer, c, epoch, step)

    print("saved", c["checkpoint"])


if __name__ == "__main__":
    main()
