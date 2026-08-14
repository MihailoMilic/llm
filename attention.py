from torch import nn
import torch
from config import device



class CausalSelfAttention(nn.Module):
    def __init__(self,C, n_heads, max_T, dropout = 0.1):
        super().__init__()
        assert C % n_heads ==0
        self.n_heads = n_heads
        self.d = C // n_heads


        self.q_k_v = nn.Linear(C, 3 * C)
        self.w_out = nn.Linear(C, C)
        self.attn_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(max_T, max_T).view(1,1,max_T, max_T)))

    def forward(self, x):
        # dimensions
        B,T,C = x.shape
        n, d = self.n_heads, self.d

        Q, K ,V = self.q_k_v(x).chunk(3, dim=-1)

        # Split C into n x d
        Q.view(B,T, n,d)
        K.view(B,T, n,d)
        V.view(B,T, n,d)

        # swap T and n, to keep iterators outside of the matmul range. We want token x entries in matmul, equiv to (B,T,C) @ (B,T,C)
        Q.transpose(1,2)
        K.transpose(1,2)
        V.transpose(1,2)

        attr = Q @ K.transpose(-2,-1) #(B,n, T,d) @ (B,n,d,T) = (B,n,T,T)
        attr = attr / d**5
        attr = attr.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attr = self.attn_drop(torch.softmax(attr, dim = -1))

        out = attr @ V

        out.transpose(1,2)
        out.contiguous()
        out.view(B,T,C)
        return self.residual_drop(self.w_out(out))









