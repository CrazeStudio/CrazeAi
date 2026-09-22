import json, torch
from torch.utils.data import DataLoader
from model import CrazeTransformer
from dataset import CrazeDataset

def main():
    with open("config.json") as f: c=json.load(f)
    dev="cuda" if torch.cuda.is_available() else "cpu"; ds=CrazeDataset("data.json",c["block_size"]); dl=DataLoader(ds,batch_size=c["batch_size"],shuffle=True)
    m=CrazeTransformer(c).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=c["learning_rate"])
    m.train()
    for ep in range(c["epochs"]):
        total=0
        for x,y in dl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); _,loss=m(x,y); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),c["grad_clip"]); opt.step(); total+=loss.item()
        print(f"epoch {ep+1}/{c['epochs']} loss={total/max(1,len(dl)):.4f}")
    torch.save({"model":m.state_dict(),"config":c},c["checkpoint"]); print("saved",c["checkpoint"])
if __name__=="__main__": main()
