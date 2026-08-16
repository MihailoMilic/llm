from torch import nn
import torch
class RoPE(nn.Module):
    def __init__(self,d, T_max):
        super().__init__()
        assert d % 2 == 0 
        theta = 1 / (10000 ** (torch.arange(0,d, 2)/d)) #(d/2,))
        print(theta.shape)
        print(theta[0])
        positions = torch.arange(T_max).float()               #(T_max,)
        angles = torch.outer(positions, theta)
        complex_table = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer("complex_table", complex_table)

    def forward(self, x):
        # x (B,n,T,d) -> (B,n,T,d)
        dtype = x.dtype
        B,n,T,d = x.shape
        x_out = x.reshape(B,n,T,d//2, 2)
        complex_table = self.complex_table[:T].reshape(1,1,T,d//2)
        x_out = torch.view_as_complex(x_out.to(torch.float32)) * complex_table
        x_out = torch.view_as_real(x_out).contiguous().view(B,n,T,d).to(dtype=dtype)
        return x_out
