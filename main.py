from attention import CausalSelfAttention
import torch



m = CausalSelfAttention(C=64, n_heads=8, max_T=128)
x = torch.randn(2, 10, 64)
print(m(x).shape)                                    # (2, 10, 64)
print(sum(p.numel() for p in m.parameters()))        # 16640
print([name for name, _ in m.named_parameters()])