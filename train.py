import math
import time
import torch
import tiktoken
import wandb

from config import device
from model import GPT

BATCH_SIZE   = 16
SEQ_LEN      = 256
MAX_STEPS    = 500
WARMUP_STEPS = 50
LR_MAX       = 3e-4
LR_MIN       = LR_MAX / 10
GRAD_CLIP    = 1.0
EVAL_EVERY   = 50

MODEL_CFG = dict(
    vocab_size = 50257,
    C          = 384,
    n_heads    = 6,
    gqa_group  = 2,
    max_T      = SEQ_LEN,
    n_layers   = 6,
    dropout    = 0.1,
)

# ── Data ───────────────────────────────────────────────────────────────────────
def load_shakespeare(enc):
    import urllib.request, os
    path = "shakespeare.txt"
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    text = open(path).read()
    tokens = enc.encode(text)
    return torch.tensor(tokens, dtype=torch.long)

def get_batch(data, batch_size, seq_len):
    # sample random starting positions
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x  = torch.stack([data[i      : i + seq_len    ] for i in ix])
    y  = torch.stack([data[i + 1  : i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)

# ── LR schedule: linear warmup + cosine decay ──────────────────────────────────
def get_lr(step):
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS          # linear ramp
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
    return LR_MIN + cosine * (LR_MAX - LR_MIN)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    wandb.init(project="llm-from-scratch", config={
        "batch_size":   BATCH_SIZE,
        "seq_len":      SEQ_LEN,
        "max_steps":    MAX_STEPS,        
        "warmup_steps": WARMUP_STEPS,
        "lr_max":       LR_MAX,
        "lr_min":       LR_MIN,
        "grad_clip":    GRAD_CLIP,
        **MODEL_CFG,
    })

    enc  = tiktoken.get_encoding("gpt2")
    data = load_shakespeare(enc)
    split = int(.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Dataset: {len(data):,} tokens  |  device: {device}")

    model = GPT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters")

    # AdamW — weight decay only on 2-D params (matrices), not biases/norms
    decay_params   = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=LR_MAX, betas=(0.9, 0.95), fused=device == "cuda")

    model.train()
    t0 = time.time()

    for step in range(MAX_STEPS):
        # update LR manually (no scheduler object — keeps it transparent)
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        # gradient clipping — returns the norm BEFORE clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        # ── logging ──────────────────────────────────────────────────────────
        wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}, step=step)

        if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            elapsed = time.time() - t0
            model.eval()
            with torch.no_grad():
                mean_val_loss = 0
                for i in range(1, 21):
                    x,y = get_batch(val_data, BATCH_SIZE, SEQ_LEN)
                    _, val_loss = model(x,y)
                    mean_val_loss = (i-1) / i * mean_val_loss  + 1 / i * val_loss.item()
                print(f"step {step:4d} | loss {loss.item():.4f} | val_loss {mean_val_loss} | "
                    f"grad_norm {grad_norm.item():.3f} | lr {lr:.2e} | {elapsed:.1f}s")
                wandb.log({"val_loss": val_loss}, step=step)
            model.train()
    torch.save(model.state_dict(), "checkpoint_500.pt")
    print("Saved checkpoint_500.pt")
    wandb.finish()

if __name__ == "__main__":
    main()
