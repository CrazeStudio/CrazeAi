"""CrazeAI's own small decoder-only transformer. No Qwen/Llama weights."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__(); inv=1.0/(base**(torch.arange(0,head_dim,2).float()/head_dim)); f=torch.outer(torch.arange(max_seq_len).float(),inv)
        self.register_buffer('cos',f.cos()[None,None],persistent=False); self.register_buffer('sin',f.sin()[None,None],persistent=False)
    def forward(self,x):
        t=x.size(2); c=self.cos[:,:,:t]; s=self.sin[:,:,:t]; a=x[...,::2]; b=x[...,1::2]; y=torch.empty_like(x)
        y[...,::2]=a*c-b*s; y[...,1::2]=a*s+b*c; return y

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); d=c['n_embd']; h=c['n_head']; kv=c['n_kv_head']; hd=d//h; self.h=h; self.kv=kv; self.hd=hd
        self.q=nn.Linear(d,h*hd,bias=False); self.k=nn.Linear(d,kv*hd,bias=False); self.v=nn.Linear(d,kv*hd,bias=False); self.o=nn.Linear(d,d,bias=False); self.rope=RoPE(hd,c['block_size']); self.dropout=c['dropout']
    def forward(self,x):
        b,t,_=x.shape; q=self.q(x).view(b,t,self.h,self.hd).transpose(1,2); k=self.k(x).view(b,t,self.kv,self.hd).transpose(1,2); v=self.v(x).view(b,t,self.kv,self.hd).transpose(1,2)
        q=self.rope(q); k=self.rope(k); r=self.h//self.kv
        if r>1: k=k.repeat_interleave(r,1); v=v.repeat_interleave(r,1)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0,is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(b,t,-1))

class SwiGLU(nn.Module):
    def __init__(self,d,f,drop):
        super().__init__(); self.g=nn.Linear(d,f,bias=False); self.u=nn.Linear(d,f,bias=False); self.d=nn.Linear(f,d,bias=False); self.drop=nn.Dropout(drop)
    def forward(self,x): return self.drop(self.d(F.silu(self.g(x))*self.u(x)))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.n1=RMSNorm(c['n_embd']); self.a=Attention(c); self.n2=RMSNorm(c['n_embd']); self.f=SwiGLU(c['n_embd'],c['ffn_dim'],c['dropout'])
    def forward(self,x):
        x=x+self.a(self.n1(x)); return x+self.f(self.n2(x))

class CrazeAI(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; d=c['n_embd']; self.tok=nn.Embedding(c['vocab_size'],d); self.blocks=nn.ModuleList([Block(c) for _ in range(c['n_layer'])]); self.norm=RMSNorm(d); self.head=nn.Linear(d,c['vocab_size'],bias=False); self.head.weight=self.tok.weight; self.apply(self._init); self._scale_residuals()
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def _scale_residuals(self):
        s=0.02/math.sqrt(2*self.c['n_layer'])
        for b in self.blocks: nn.init.normal_(b.a.o.weight,0,s); nn.init.normal_(b.f.d.weight,0,s)
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
    def forward(self,input_ids,labels=None):
        if input_ids.size(1)>self.c['block_size']: raise ValueError('sequence exceeds block_size')
        x=self.tok(input_ids)
        for b in self.blocks: x=b(x)
        logits=self.head(self.norm(x)); loss=None
        if labels is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),ignore_index=-100)
        return logits,loss
