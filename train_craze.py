"""Train CrazeAI from scratch. No pretrained model is loaded.
Requires: torch sentencepiece
Example:
  python train_craze.py --config craze_config.json --data data.json --tokenizer tokenizer.model
"""
import argparse, json, math, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from craze_model import CrazeAI


def load_text(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    entries=obj.get('entries',obj if isinstance(obj,list) else [])
    out=[]
    for e in entries:
        if isinstance(e,dict):
            q=str(e.get('instruction','')).strip(); inp=str(e.get('input','')).strip(); ans=str(e.get('output','')).strip()
            if q: out.append(f'<|user|>\n{q}\n{inp}\n<|assistant|>\n{ans}<|end|>')
        elif str(e).strip(): out.append(str(e))
    return '\n'.join(out)

class Packed(Dataset):
    def __init__(self, ids, seq):
        n=(len(ids)-1)//seq
        self.x=torch.tensor(ids[:n*seq],dtype=torch.long).view(n,seq)
        self.y=torch.tensor(ids[1:n*seq+1],dtype=torch.long).view(n,seq)
    def __len__(self): return self.x.size(0)
    def __getitem__(self,i): return self.x[i],self.y[i]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='craze_config.json'); ap.add_argument('--data',default='data.json'); ap.add_argument('--tokenizer',required=True); ap.add_argument('--out',default='checkpoints/crazeai_base.pt'); a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text()); seed=cfg['training']['seed']; random.seed(seed); torch.manual_seed(seed)
    sp=spm.SentencePieceProcessor(model_file=a.tokenizer); text=load_text(a.data); ids=[1]+sp.encode(text,out_type=int)+[2]
    ds=Packed(ids,cfg['block_size']);
    if len(ds)==0: raise RuntimeError('Dataset is too small for one training sequence. Add real corpus data before training.')
    loader=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=True,drop_last=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=CrazeAI(cfg).to(device); print(f'CrazeAI parameters: {model.parameter_count():,}'); print('device:',device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'],betas=(0.9,0.95))
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    model.train(); step=0; accum=cfg['training']['gradient_accumulation']; max_steps=cfg['training']['max_steps']
    while step<max_steps:
        for x,y in loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    _,loss=model(x,y); loss=loss/accum
                scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']); scaler.step(opt); scaler.update()
            step+=1
            if step%50==0: print(f'step={step} loss={loss.item()*accum:.4f}')
            if step%1000==0:
                Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'config':cfg,'tokenizer':a.tokenizer,'step':step},a.out); print('checkpoint:',a.out)
            if step>=max_steps: break
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'config':cfg,'tokenizer':a.tokenizer,'step':step},a.out)

if __name__=='__main__': main()
