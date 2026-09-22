import torch

class ByteTokenizer:
    vocab_size = 256
    bos = 2
    eos = 3
    def encode(self, text, add_bos=True, add_eos=True):
        x=list(text.encode("utf-8", errors="replace"))
        if add_bos: x=[self.bos]+x
        if add_eos: x=x+[self.eos]
        return x
    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")
    def tensor(self, text):
        return torch.tensor(self.encode(text), dtype=torch.long)
