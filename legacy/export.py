import argparse
import struct
import os
import torch

MAGIC = b"CRAZEAI1"
VERSION = 1

def u32(f, x):
    f.write(struct.pack("<I", int(x)))

def u64(f, x):
    f.write(struct.pack("<Q", int(x)))

def export(checkpoint, output):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"]
    state = ck["model"]
    tensors = []
    for name, tensor in state.items():
        if name == "head.weight" or name.endswith(".mask"):
            continue
        t = tensor.detach().cpu().contiguous().float()
        tensors.append((name, t))

    with open(output, "wb") as f:
        f.write(MAGIC)
        u32(f, VERSION)
        u32(f, c["vocab_size"])
        u32(f, c["block_size"])
        u32(f, c["n_embd"])
        u32(f, c["n_layer"])
        u32(f, c["n_head"])
        u32(f, len(tensors))
        for name, t in tensors:
            name_b = name.encode("utf-8")
            raw = t.numpy().tobytes(order="C")
            u32(f, len(name_b))
            f.write(name_b)
            u32(f, t.ndim)
            for d in t.shape:
                u32(f, d)
            u64(f, len(raw))
            f.write(raw)

    print(f"exported {output}: {os.path.getsize(output) / 1048576:.1f} MiB")
    print(f"tensors: {len(tensors)}")
    print("dtype: FP32, format: CRAZEAI1")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="crazeai.pt")
    p.add_argument("--output", default="crazeai.craze")
    a = p.parse_args()
    export(a.checkpoint, a.output)
