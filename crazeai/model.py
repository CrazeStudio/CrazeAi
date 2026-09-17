import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__(); assert d_model % n_heads == 0
        self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.qkv=nn.Linear(d_model,3*d_model,bias=False); self.proj=nn.Linear(d_model,d_model,bias=False)
    def forward(self,x,past_kv=None):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.n_heads,self.head_dim).transpose(1,2); k=k.view(b,t,self.n_heads,self.head_dim).transpose(1,2); v=v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        if past_kv is not None:
            pk,pv=past_kv; k=torch.cat((pk,k),dim=2); v=torch.cat((pv,v),dim=2)
        total=k.size(2)
        if t==1 and past_kv is not None:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        else:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c)),(k,v)

class MLP(nn.Module):
    def __init__(self,d_model,ffn_dim):
        super().__init__(); self.fc1=nn.Linear(d_model,ffn_dim,bias=False); self.fc2=nn.Linear(ffn_dim,d_model,bias=False)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self,d_model,n_heads,ffn_dim):
        super().__init__(); self.ln1=nn.LayerNorm(d_model); self.attn=CausalSelfAttention(d_model,n_heads); self.ln2=nn.LayerNorm(d_model); self.mlp=MLP(d_model,ffn_dim)
    def forward(self,x,past=None):
        a,kv=self.attn(self.ln1(x),past); return x+a+self.mlp(self.ln2(x+a)),kv

class NanoSLM(nn.Module):
    def __init__(self,vocab_size=4096,d_model=128,n_heads=4,n_layers=4,ffn_dim=512,ctx_len=256):
        super().__init__(); self.ctx_len=ctx_len
        self.tok_emb=nn.Embedding(vocab_size,d_model); self.pos_emb=nn.Embedding(ctx_len,d_model)
        self.blocks=nn.ModuleList([Block(d_model,n_heads,ffn_dim) for _ in range(n_layers)]); self.ln_f=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab_size,bias=False); self.head.weight=self.tok_emb.weight
    def forward(self,idx,past_kv=None):
        b,t=idx.shape
        past_len=0 if past_kv is None else past_kv[0][0].size(2)
        if past_len+t>self.ctx_len: raise ValueError("context window exceeded")
        pos=torch.arange(past_len,past_len+t,device=idx.device)
        x=self.tok_emb(idx)+self.pos_emb(pos)
        new=[]
        for i,block in enumerate(self.blocks): x,kv=block(x,None if past_kv is None else past_kv[i]); new.append(kv)
        return self.head(self.ln_f(x)),tuple(new)

def load_model(path,device="cpu"):
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    model=NanoSLM(); checkpoint=torch.load(path,map_location=device,weights_only=False); state=checkpoint["model"] if isinstance(checkpoint,dict) and "model" in checkpoint else checkpoint; model.load_state_dict(state); return model.to(device).eval()

@torch.inference_mode()
def generate(model,tokenizer,prompt,max_new_tokens=64,temperature=0.8,top_k=40,top_p=0.95):
    encoded=tokenizer.encode(prompt); ids=encoded.ids if hasattr(encoded,"ids") else encoded
    if not ids: ids=[0]
    device=next(model.parameters()).device; ids=ids[-model.ctx_len:]
    x=torch.tensor([ids],dtype=torch.long,device=device)
    logits,past=model(x)
    for _ in range(max_new_tokens):
        logits=logits[:,-1,:]
        if temperature<=0: next_id=logits.argmax(-1,keepdim=True)
        else:
            logits=logits/temperature
            if top_k:
                k=min(top_k,logits.size(-1)); threshold=torch.topk(logits,k,dim=-1).values[:,-1:]; logits=logits.masked_fill(logits<threshold,-float("inf"))
            if top_p<1:
                sl,si=torch.sort(logits,descending=True); probs=F.softmax(sl,dim=-1); cum=torch.cumsum(probs,dim=-1); remove=cum>top_p; remove[...,1:]=remove[...,:-1].clone(); remove[...,0]=False; sl=sl.masked_fill(remove,-float("inf")); logits=torch.zeros_like(logits).scatter(1,si,sl)
            next_id=torch.multinomial(F.softmax(logits,dim=-1),1)
        x=torch.cat((x,next_id),dim=1)
        logits,past=model(next_id,past)
    return tokenizer.decode(x[0].tolist())
