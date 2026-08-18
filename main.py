from attention import CausalSelfAttention, SwiGLU
import tiktoken
import torch
from embeddings import RoPE
from model import Block, GPT, generate

model = GPT(vocab_size=50257, C=384, n_heads=6, gqa_group=2, max_T=256, n_layers=6)
# idx = torch.randint(0, 50257, (2, 16))
# logits, loss = m(idx[:, :-1], idx[:, 1:])
# print(logits.shape, loss.item())
# print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")



enc = tiktoken.get_encoding("gpt2")
prompt = torch.tensor([enc.encode("To be or not to be")], dtype=torch.long)


import time
out = []
for flag in (False, True):
    torch.manual_seed(0)
    model.clear_cache()
    t0 = time.time()
    o = generate(model, prompt, 200, use_cache=flag)
    print(flag, 200 / (time.time() - t0), "tok/s")
    out.append(o)

print(torch.equal(out[0], out[1]))
