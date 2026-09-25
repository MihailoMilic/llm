# llm

A LLaMA-style decoder-only transformer written from scratch in PyTorch, with hand-written fused Triton
kernels for its memory-bound operations.
## Layout

| File | |
|---|---|
| `attention.py` | Causal self-attention generalised to grouped-query attention, plus SwiGLU. Includes `uptrain_to_gqa`, which pools K/V weights to convert a trained model to fewer KV groups. |
| `embeddings.py` | RoPE: PyTorch reference, Triton forward and backward kernels, and the `TritonRoPE` autograd module. |
| `normalisation.py` | RMSNorm: PyTorch reference (fp32 reduction, cast back) and a fused forward kernel, in progress. |
| `model.py` | Decoder block, full model, and `generate` with optional KV cache. |
| `train.py` | Training loop: AdamW with decay on 2-D parameters only, linear warmup and cosine decay, gradient clipping, wandb logging. |
| `main.py` | KV-cache benchmark: tokens/sec with and without the cache, asserting both produce identical tokens. |
| `test_rope.py` | Test suite for the RoPE kernels (see below). |
| `fused-softmax.py`, `vector-add.py` | The official Triton tutorials, worked through as fundamentals. Not my own kernels. |

## What is verified

```bash
pytest test_rope.py        # needs CUDA
```

The RoPE forward and backward kernels are checked against the PyTorch reference across batch/head/sequence
shapes, head dimensions of 2, 64 and 96, fp32 and bf16, three position offsets, and five memory layouts,
including non-contiguous inputs and gradients with zero strides.

Two of the tests do not use the reference at all:

- **Adjoint:** `<Rx, g> == <x, Rᵀg>`, which holds only if the backward kernel is the exact transpose of the forward.
- **Orthogonality:** `RᵀR x == x`, since a rotation is its own inverse under transpose.

Writing these found a bug in the *reference* implementation, not the kernel: the complex view was being taken
on a non-contiguous tensor, which is exactly what `attention.py` passes in after transposing heads and time.

## In progress

- **Fused RMSNorm.** Forward drafted, not yet passing. The backward needs a two-stage reduction for the weight
  gradient, since every row contributes to the same weight vector, validated by `gradcheck` in float64.

## Known limitation

Both the RoPE kernel and the tutorial softmax set `BLOCK_SIZE = next_power_of_2(row_length)` and process a row
in a single block. The row therefore has to fit in registers and shared memory. For RoPE this never binds as
the row is `head_dim / 2`, at most 64. For a softmax over a sequence or vocabulary dimension it does: a 50k-wide
row in fp32 is ~200KB, past the SRAM of any SM, and the kernel would spill to local memory, which is backed by
DRAM and defeats the purpose of fusing. The fix is an online softmax keeping a running
max and a rescaled running sum — which is the next thing to write.

## Next

1. Streaming reductions, lifting the single-block row assumption.
2. Chunked cross-entropy, computing the loss over vocabulary chunks so the `(B, T, 50257)` logits tensor —
   the largest allocation in the model — is never materialised. The result to report there is peak VRAM and
   maximum batch size before OOM, not speed.
3. Profiling each kernel against a `torch.compile`'d baseline, reporting isolated op speedup, that op's share
   of step time, and the resulting end-to-end change together.
4. A weight-only int8/fp8 dequantise-fused path, with per-channel error analysis.

## Running it

```bash
conda create -n llm python=3.11 -y && conda activate llm
pip install "torch==2.2.2" "numpy<2" triton tiktoken matplotlib wandb pytest

python train.py    # 500 steps on tinyshakespeare, ~11M params, 6 layers, d=384, 6 heads, 2 KV groups
python main.py     # KV-cache tokens/sec comparison
```
