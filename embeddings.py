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
def RoPE_kernel(x_ptr, out_ptr,bx_stride, nhx_stride, tx_stride, bo_stride, nho_stride, to_stride,n,T,d, d_half, offset,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)

    t = pid % T
    nh = ( pid // T ) % n
    b = pid // (T * n)
    p = t + offset

    row_ptr = x_ptr + b * bx_stride + nh * nhx_stride + t * tx_stride
    row_out_ptr = out_ptr + b * bo_stride + nh * nho_stride + t * to_stride

    offs = tl.arange(0, BLOCK_SIZE)

    re_offs = 2 * offs 
    im_offs = 2 * offs + 1
    mask = offs < d_half
    re = tl.load(row_ptr + re_offs, mask = mask, other = 0.0)
    im = tl.load(row_ptr + im_offs, mask = mask, other = 0.0)
    theta = 1 / tl.exp((offs.to(tl.float32) * 2.0 / d ) * 9.210340371976184)
    a = p * theta
    cos = tl.cos(a)
    sin = tl.sin(a)
    out_re = re * cos - im * sin
    out_im = re * sin + im * cos
    tl.store(row_out_ptr + re_offs, out_re, mask = mask)
    tl.store(row_out_ptr + im_offs, out_im, mask = mask)
@triton.jit
def RoPE_backward_kernel(g_ptr, dx_ptr,gx_stride,nhg_stride,tg_stride, bdx_stride,nhdx_stride,tdx_stride,n, T, d, d_half, offset, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0)

    t = pid % T
    nh = ( pid // T ) % n
    b = pid // (T * n)
    p = t + offset

    g_row_ptr = g_ptr + b * gx_stride + nh * nhg_stride + t * tg_stride
    dx_row_ptr = dx_ptr + b * bdx_stride + nh * nhdx_stride + t * tdx_stride
   
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < d_half
    re_offs = 2 * offs
    im_offs = 2 * offs + 1
    g_re = tl.load(g_row_ptr + re_offs, mask = mask, other = 0.0).to(tl.float32)
    g_im = tl.load(g_row_ptr + im_offs, mask = mask, other = 0.0).to(tl.float32)
    theta = 1 / tl.exp((offs.to(tl.float32) * 2.0 / d ) * 9.210340371976184)
    a = p * theta
    cos = tl.cos(a)
    sin = tl.sin(a)
    out_re = g_re * cos + g_im * sin
    out_im = - g_re * sin + g_im * cos
    tl.store(dx_row_ptr + re_offs, out_re, mask=mask)
    tl.store(dx_row_ptr + im_offs, out_im, mask=mask)


def rope_forward(x, offset):
    B, n, T, d = x.shape
    # x = x.contiguous() outer dims having arbitrary stride handled, still assuming stride of -1 is 1.
    if x.stride()[-1] != 1: 
        print("We expect transpose of only (-2,-3), last dim is expect to have stride 1")
        x = x.contiguous()
    bx_stride, nhx_stride, tx_stride, _ = x.stride()
    out = torch.empty_like(x)
    bo_stride, nho_stride, to_stride, _ = out.stride()
    d_half = d // 2
    BLOCK_SIZE = triton.next_power_of_2(d_half)
    RoPE_kernel[(B * n * T,)](
        x, out,bx_stride, nhx_stride, tx_stride,bo_stride, nho_stride, to_stride,n,T, d, d_half, offset, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
def rope_backward(g, offset):
    B, n, T, d = g.shape
    if g.stride()[-1] != 1:
        g = g.contiguous()
    dx = torch.empty_like(g)
    # our tl.arange gathers across the last axis, therefore it contiguous() is less costly than fetching the last dim ourselves
    gx_stride, nhg_stride, tg_stride, _ = g.stride()

    bdx_stride, nhdx_stride, tdx_stride, _ = dx.stride()
    d_half = d // 2
    BLOCK_SIZE = triton.next_power_of_2(d_half)
    # g_ptr, dx_ptr,gx_stride,nhg_stride,tg_stride, bdx_stride,nhdx_stride,tdx_stride,n,T, d, d_half, offset, BLOCK_SIZE:tl.constexpr
    RoPE_backward_kernel[(B * n * T,)](
        g, dx,
        gx_stride,nhg_stride,tg_stride, 
       bdx_stride,nhdx_stride,tdx_stride,
        n, T, d, d_half, offset,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx


class _RoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, offset):
        ctx.offset = offset
        return rope_forward(x, offset)
    @staticmethod
    def backward(ctx, g):
        return rope_backward(g, ctx.offset), None 

class TritonRoPE(nn.Module):
    def __init__(self, d, T_max):
        super().__init__()
        assert d % 2 == 0
        self.d = d
        self.T_max = T_max
    def forward(self, x, offset = 0):
        assert x.shape[-1] == self.d
        assert offset + x.shape[-2] <= self.T_max
        return _RoPEFn.apply(x, offset)



