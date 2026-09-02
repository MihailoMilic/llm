import torch
from torch import nn
import triton
class RMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(C))

    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms * self.weight).to(dtype)
    

