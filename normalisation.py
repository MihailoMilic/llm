import torch
from torch import nn
import triton
import triton.language as tl
class RMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(C))

    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt() #rsqrt is reciprocal square root
        return (x_fp32 * rms * self.weight).to(dtype)

#element wise pow of 2, last row reduction into mean add eps divide across last row of original x to 


@triton.jit
def rmsn_forward_kernel(x_ptr, out_ptr, w_ptr, x_row_stride, out_row_stride, n_rows, n_cols, eps, BLOCK_SIZE:tl.constexpr, num_stages: tl.constexpr):
    program_start = tl.program_id(0)
    program_step  = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    weight_ptr = w_ptr + col_offsets
    mask = col_offsets < n_cols
    weight = tl.load(weight_ptr, mask = mask, other = 0.0).to(tl.float32)

    for row_idx in tl.range(program_start, n_rows, program_step, num_stages = num_stages):

        row_start_ptr = x_ptr + row_idx * x_row_stride
        out_row_start_ptr = out_ptr + row_idx * out_row_stride
                
        row_ptrs = row_start_ptr + col_offsets
        out_row_ptrs = out_row_start_ptr + col_offsets

        x_row = tl.load(row_ptrs, mask = mask, other = 0.0).to(tl.float32)
        
        row_rms = tl.rsqrt(1 / n_cols * tl.sum(x_row * x_row) + eps)
        x_rmsnorm = x_row * row_rms * weight 

        tl.store(out_row_ptrs, x_rmsnorm, mask = mask)


