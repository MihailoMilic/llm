import torch
import torch.nn.functional as F
from torch import nn
from attention import CausalSelfAttention, SwiGLU
from normalisation import RMSNorm
class Block(nn.Module):
    def __init__(self, C, n_heads, gqa_group, max_T, dropout=0.1):
            super().__init__()  
            self.norm = RMSNorm(C)
            self.attn = CausalSelfAttention(C, n_heads, gqa_group, max_T, dropout=0.1)
            self.norm_mlp = RMSNorm(C)
            self.mlp = SwiGLU(C)
    def forward(self,x, use_cache = False):

         x = x + self.attn(self.norm(x), use_cache = use_cache)
        # postnorm
         x = x + self.mlp(self.norm_mlp(x))
         return x

class GPT(nn.Module):
    def __init__(self, vocab_size, C, n_heads, gqa_group, max_T, n_layers, dropout=0.1):
        super().__init__()
        self.max_T = max_T
        self.tok_emb = nn.Embedding(vocab_size, C)
        self.blocks = nn.ModuleList([
            Block(C, n_heads, gqa_group, max_T, dropout) for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(C)
        self.lm_head = nn.Linear(C, vocab_size, bias=False)

    def clear_cache(self):
        for block in self.blocks:
            block.attn.clear_cache()

    def forward(self, idx, targets=None,*,  use_cache = False):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, use_cache)
        x = self.norm_out(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss


@torch.no_grad()
def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None, use_cache = False):
    
    was_training = model.training
    model.eval()
    if use_cache:
        model.clear_cache()
    for i in range(max_new_tokens):
        if use_cache:
            prompt_max = prompt if i ==0 else prompt[:,-1:]
        else:
            prompt_max = prompt[:, -model.max_T:] # slices the prompt to fit the max T

        logits, _ = model(prompt_max,targets = None ,use_cache = use_cache) #returns logits and loss
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float('-inf'))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        prompt = torch.cat([prompt, next_id], dim=1)

    if was_training:
        model.train() 
    return prompt