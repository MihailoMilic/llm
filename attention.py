from torch import nn
import torch.nn.functional as F
import torch
from config import device
from embeddings import RoPE


class CausalSelfAttention(nn.Module):
    def __init__(self,C, n_heads,gqa_group, max_T, dropout = 0.1):
        super().__init__()
        assert C % n_heads ==0
        assert n_heads % gqa_group == 0
        self.C = C
        self.gqa_group = gqa_group

        self.n_heads = n_heads
        self.d = C // n_heads

        self.rope = RoPE(self.d, max_T)

        self.q_k_v = nn.Linear(C, C + 2*(self.d *self.gqa_group))
        self.w_out = nn.Linear(C, C)
        self.attn_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(max_T, max_T).view(1,1,max_T, max_T)))

        self.register_buffer("k_cache",None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def uptrain_to_gqa(self, new_gqa_group):

        assert self.gpa_group % new_gqa_group == 0

        W = self.q_k_v.weight.data
        b = self.q_k_v.bias.data if self.q_k_v.bias is not None else None

        qW,kW,vW = W.split([self.C, self.d*self.gqa_group, self.d*self.gqa_group], dim=0)
        qb,kb, vb = b.split([self.C, self.d*self.gqa_group, self.d*self.gqa_group], dim=0) if b is not None else None, None, None

        group_size = self.gqa_group // new_gqa_group
        def pool(t):
            t = t.view(new_gqa_group, group_size,self.d, *t.shape[1:]) # (C', T, B) -> (new_g, g_size, d, T, B) note that C' = new_g * g_size * d
            return t.mean(dim=1).reshape(new_gqa_group * self.d, *t.shape[1:]) #(new_g, g_size, d, T, B) -> (new_g, 1, d, T, B)

        new_q_k_v = nn.Linear(self.C, self.C + self.d * new_gqa_group, bias= b is not None, device = W.device, dtype=W.dtype)

        new_q_k_v.weight.data = torch.concat([qW, pool(kW), pool(vW)], dim = 0)
        if b is not None:
            assert qb != None
            new_q_k_v.bias.data = torch.concat([qb, pool(kb), pool(vb)], dim = 0)
        self.q_k_v = new_q_k_v
        self.gqa_group = new_gqa_group


    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache = False):

        # dimensions
        B,T,C = x.shape
        n, d, g = self.n_heads, self.d, self.gqa_group
        p = n // g

        Q, K ,V = self.q_k_v(x).split([C, self.d*self.gqa_group, self.d*self.gqa_group], dim=-1)
        
        # Split C into n x d
        Q = Q.view(B,T, n,d)
        K = K.view(B,T, g,d)
        V =V.view(B,T, g,d)
        # swap T and n, to keep iterators outside of the matmul range. We want token x entries in matmul, equiv to (B,T,C) @ (B,C,T)
        Q = Q.transpose(1,2) #(B,n,T,d)
        K = K.transpose(1,2) #(B,g,T,d)
        V = V.transpose(1,2) #(B,g,T,d)

        past_len = self.k_cache.shape[-2] if (use_cache and self.k_cache is not None) else 0
        Q = self.rope(Q, offset = past_len)
        K = self.rope(K, offset = past_len)
       
        if use_cache:
            if self.k_cache is not None:
                if self.v_cache is not None:
                    K = torch.concat([self.k_cache, K], dim = -2)
                    V = torch.concat([self.v_cache, V], dim = -2)
            self.k_cache = K
            self.v_cache = V
        T_full = K.shape[-2]

        K = K.unsqueeze(2).expand(B,g,p,T_full,d).reshape(B,n,T_full,d)     #(B,g,T,d) -> (B,g,1,T,d) -> (B,g,p,T,d) -> (B,g*p,T,d) = (B,n,T,d)
        V = V.unsqueeze(2).expand(B,g,p,T_full,d).reshape(B,n,T_full,d)     #(B,g,T,d) -> (B,g,1,T,d) -> (B,g,p,T,d) -> (B,g*p,T,d) = (B,n,T,d)
        attr = Q @ K.transpose(-2,-1) #(B,n, T,d) @ (B,n,d,T) = (B,n,T,T)
        attr = attr / d**0.5
        attr = attr.masked_fill(self.mask[:, :, past_len:T+past_len, :T_full] == 0, float('-inf'))
        attr = self.attn_drop(torch.softmax(attr, dim = -1))

        out = attr @ V #(B,n,T,T) @(B,n,T,d) = (B,n,T,d)

        out=  out.transpose(1,2).contiguous().view(B,T,C) #(B,n,T,d) -> (B,T,n,d) -> (B,T,C)
        return self.residual_drop(self.w_out(out)) # out (B,T,C)


class SwiGLU(nn.Module):
    def __init__(self, C, hidden = None, multiple_of = 64):
        super().__init__()
        self.hidden = int(multiple_of * ((2/3 * 4 * C + multiple_of -1) // multiple_of) if hidden is None else hidden)

        self.v_w = nn.Linear(C, 2* self.hidden, bias = False)
        self.resid_width = nn.Linear(self.hidden, C, bias=False)

    def forward(self,x):
        v_out,w_out = self.v_w(x).chunk(2, dim = -1)
        w_down = self.resid_width((F.silu(w_out) * v_out))
        return w_down

        





