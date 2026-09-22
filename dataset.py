import json, torch
from torch.utils.data import Dataset
from tokenizer import ByteTokenizer

class CrazeDataset(Dataset):
    def __init__(self,path,block_size):
        self.tok=ByteTokenizer(); self.block=block_size
        with open(path,encoding="utf-8") as f: data=json.load(f)["entries"]
        self.samples=[]
        for e in data:
            text=f"### Instruction:\n{e.get('instruction','')}\n### Input:\n{e.get('input','')}\n### Output:\n{e.get('output','')}"
            ids=self.tok.encode(text)
            if len(ids)>block_size+1: ids=ids[:block_size+1]
            self.samples.append(torch.tensor(ids,dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        x=self.samples[i]; y=x.clone();
        if len(x)<self.block: x=torch.cat([x,torch.zeros(self.block-len(x),dtype=torch.long)])
        if len(y)<self.block: y=torch.cat([y,torch.zeros(self.block-len(y),dtype=torch.long)])
        return x[:self.block],y[:self.block]
