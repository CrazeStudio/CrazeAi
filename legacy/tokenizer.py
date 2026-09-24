import torch

class ByteTokenizer:
    """UTF-8 byte tokenizer with two reserved special IDs."""
    vocab_size = 258
    bos = 256
    eos = 257

    def encode(self, text, add_bos=True, add_eos=True):
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids.insert(0, self.bos)
        if add_eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids):
        data = bytes(i for i in ids if 0 <= i < 256)
        return data.decode("utf-8", errors="replace")

    def tensor(self, text):
        return torch.tensor(self.encode(text), dtype=torch.long)
