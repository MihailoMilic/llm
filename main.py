from attention import CausalSelfAttention, SwiGLU
import torch
from embeddings import RoPE

r = RoPE(d=64, T_max=128)
qv, kv = torch.randn(64), torch.randn(64)
q = qv.expand(1,1,128,64).contiguous()   # same vector at every position
k = kv.expand(1,1,128,64).contiguous()
qr, kr = r(q), r(k)

m, n, s = 5, 12, 40
print(qr[0,0,m] @ kr[0,0,n], qr[0,0,m+s] @ kr[0,0,n+s])