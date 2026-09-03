from torch import nn
import torch
import math
import triton.language as tl
import triton
class RoPE(nn.Module):
    def __init__(self,d, T_max):
        super().__init__()
        assert d % 2 == 0 
        theta = 1 / (10000 ** (torch.arange(0,d, 2)/d)) #(d/2,)) 

        positions = torch.arange(T_max).float()               #(T_max,)
        angles = torch.outer(positions, theta)
        complex_table = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer("complex_table", complex_table)

    def forward(self, x, offset):
        # x (B,n,T,d) -> (B,n,T,d)
        dtype = x.dtype
        B,n,T,d = x.shape
        x_out = x.reshape(B,n,T,d//2, 2)
        complex_table = self.complex_table[offset:offset+T].reshape(1,1,T,d//2)
        x_out = torch.view_as_complex(x_out.to(torch.float32)) * complex_table
        x_out = torch.view_as_real(x_out).contiguous().view(B,n,T,d).to(dtype=dtype)
        return x_out




@triton.jit
def RoPE_kernel(x_ptr, out_ptr, T,d, d_half, offset,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)
    t = pid % T
    p = t + offset
    row_ptr = x_ptr + pid * d
    row_out_ptr = out_ptr + pid * d
    offs = tl.arange(0, BLOCK_SIZE)
    re_offs = 2 * offs 
    im_offs = 2 * offs + 1
    mask = offs < d_half
    re = tl.load(row_ptr + re_offs, mask = mask, other = 0.0)
    im = tl.load(row_ptr + im_offs, mask = mask, other = 0.0)
    theta = 1 / tl.exp((offs.to(tl.float32) * 2.0 / d) * 9.210340371976184)
    a = p * theta
    cos = tl.cos(a)
    sin = tl.sin(a)
    out_re = re * cos - im * sin
    out_im = re * sin + im * cos
    tl.store(row_out_ptr + re_offs, out_re, mask = mask)
    tl.store(row_out_ptr + im_offs, out_im, mask = mask)




def rope_forward(x, offset):
    B, n, T, d = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    d_half = d // 2
    BLOCK_SIZE = triton.next_power_of_2(d_half)
    RoPE_kernel[(B * n * T,)](
        x, out, T, d, d_half, offset, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


torch.manual_seed(0)
device = 'cuda'
x = torch.randn(2, 4, 16, 64, device=device)
offset = 0
r = RoPE(64, 128).to(device)
x_to = r.forward(x, offset= offset)
x_t = rope_forward(x, offset= offset)
print(torch.allclose(x_to, x_t, atol=1e-5, rtol=1e-5))
print((x_to - x_t).abs().max().item())
