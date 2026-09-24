"""Final training entrypoint for an independently initialized CrazeAI model."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from craze_model import CrazeAI

class DS(Dataset):
    def __init__(self,path,tokenizer,block):
        self.rows=[]; self.tok=tokenizer; self.block=block
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            x=json.loads(line); text=''
            for m in x['messages']: text += f"<|{m['role']}|>\n{m['content']}\n"
            ids=tokenizer.encode(text, out_type=int)[:block]
            if len(ids)>2: self.rows.append(ids)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        x=torch.tensor(self.rows[i],dtype=torch.long); y=x.clone(); return x,y

def collate(batch,pad):
    n=max(len(x[0]) for x in batch); xs=[]; ys=[]
    for x,y in batch:
        z=n-len(x); xs.append(torch.cat([x,torch.full((z,),pad,dtype=torch.long)])); ys.append(torch.cat([y,torch.full((z,),-100,dtype=torch.long)]))
    return torch.stack(xs),torch.stack(ys)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/craze_train.jsonl'); ap.add_argument('--tokenizer',default='tokenizer.model'); ap.add_argument('--out',default='crazeai.pt'); ap.add_argument('--steps',type=int,default=10000); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--lr',type=float,default=3e-4); args=ap.parse_args()
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=args.tokenizer)
    cfg=json.load(open('craze_config.json')); cfg['vocab_size']=sp.vocab_size()
    model=CrazeAI(cfg); device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    ds=DS(args.data,sp,cfg['block_size']); dl=DataLoader(ds,batch_size=args.batch,shuffle=True,collate_fn=lambda b:collate(b,sp.pad_id() if sp.pad_id()>=0 else 0))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.1); step=0; model.train()
    while step<args.steps:
        for x,y in dl:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); _,loss=model(x,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
            if step%100==0: print(f'step={step} loss={loss.item():.4f} ppl={math.exp(min(loss.item(),20)):.2f}')
            if step>=args.steps: break
    torch.save({'model':model.state_dict(),'config':cfg,'steps':step},args.out); print('saved',args.out)

if __name__=='__main__': main()
