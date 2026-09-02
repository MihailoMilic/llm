# Roadmap: From-Scratch LLM + Custom Triton Kernels

**Project:** One repository. A decoder-only LLM built from scratch in raw PyTorch, then made measurably faster with hand-written fused GPU kernels.

**Narrative:** _I built it, then I made it fast._

**Target audience for the artifact:** researchers and engineers at Mistral, Kyutai, FAIR Paris, and academic groups (CMAP, INRIA) doing training/inference efficiency work.

---

## Status

|Phase|Days|Dates|State|
|---|---|---|---|
|Week 1 — Architecture|1–7|Aug 12–18|In progress (Day 7 today)|
|Phase 2 — Triton kernels|8–17|Aug 19–31|Planned|
|Phase 3 — Packaging & launch|18–21|Sep 1–4|Planned|

---

## Week 1 — Core Architecture (Days 1–7)

**Goal:** modern decoder-only LLM from scratch. No Hugging Face abstractions.

- **Day 1** — Causal Multi-Head Attention. Tensor shape tracing, `torch.tril` mask. ✅
- **Day 2** — SwiGLU MLP + Rotary Positional Embeddings, implemented manually. ✅
- **Day 3** — Grouped-Query Attention. KV head expansion via `repeat_interleave`. ✅
- **Day 4** — RMSNorm (hand-written, `torch` ops) + full `Llama3StyleDecoder` assembly, pre-norm. ✅
- **Day 5** — `tiktoken` BPE, custom `DataLoader` yielding shifted `(input_ids, target_ids)`. ✅
- **Day 6** — KV-cache with pre-allocated buffer. Benchmark tokens/sec with vs. without. ✅
- **Day 7 (today, Aug 18)** — Training loop: AdamW, LR warmup, cosine decay, gradient clipping. `wandb` logging for loss, grad norm, LR. First 500-step run.

> **Load-bearing:** the training loop is what makes end-to-end kernel benchmarks possible. It is not a leftover from the old plan.

### Carry forward from Week 1

- Keep the hand-written `RMSNorm` as the **reference implementation**. It is the correctness oracle for the kernel.
- Cast to fp32 inside the norm computation (`x.float()`, compute, cast back). Triton's `tl.sum` accumulates in fp32 by default — if the PyTorch reference doesn't match, `allclose` will fail for reasons that aren't your kernel's fault.
- Note the eps convention: `mean(x²) + eps` then `rsqrt`, matching Llama. Replicate exactly in the kernel.

---

## Phase 2 — Triton Kernels (Days 8–17, Aug 19–31)

**Goal:** systematic fusion pass over the memory-bound ops in the model, with honest measurement of what each one buys.

### The thesis

Matmuls run near hardware peak — cuBLAS is already at the bandwidth/compute limit and there is nothing to reclaim. The _glue between them_ — normalization, positional rotation, activations, loss — is memory-bound: each PyTorch op reads from HBM, computes, writes back. Fusion loads once into SRAM, does everything there, writes once.

Two distinct payoffs, demonstrated by two different kernels:

- **Bandwidth** (RMSNorm, RoPE, SwiGLU): eliminate redundant HBM round-trips → speed.
- **Capacity** (chunked cross-entropy): never materialize the intermediate at all → memory.

### Infrastructure

- **GPU:** rented RTX 4090 on RunPod or Vast.ai (~$0.35/hr). **Stop the pod when not working** — billing is per-hour of uptime, disk persists for pennies. Realistic total: **$25–50** for the whole phase.
- **Workflow:** repo on GitHub, cloned on the pod, edited via VS Code Remote-SSH. Persistent volume so the environment survives between sessions.
- Optionally use free Colab (T4) for Days 8–9 tutorials only. A T4 is fine for learning, wrong for benchmarking.
- Do **not** plan to "write offline and test later." Kernel work is a tight edit-run-verify loop; batched bugs are exponentially harder than sequential ones.

### Ordering principle

Increasing difficulty, so a partial result always ships.

---

### Day 8–9 (Aug 19–20) — Triton fundamentals

Official tutorials: vector add, then fused softmax.

What you are actually buying: the **block-level mental model**. `tl.program_id`, block pointers, masking for non-power-of-two dimensions, `tl.constexpr`, autotuning. Skipping this to get to "your" kernel is the standard way to lose four days later.

**Do not skip. Do not skim.**

---

### Day 10 (Aug 21) — Fused RoPE (forward + backward)

Easiest backward pass you will write: rotation is orthogonal, so the gradient is the inverse rotation — the same kernel with negated angles.

- Fuse cos/sin lookup + pairwise rotation into one kernel, in place.
- Verify against your Day 2 implementation with `torch.allclose`.
- Purpose: learn the workflow on a problem where the math cannot bite you.

---

### Day 11 (Aug 22) — Fused RMSNorm, forward

Row-wise reduction — the canonical Triton shape.

- Load the row once, compute mean-square → rsqrt → scale → weight multiply, all in SRAM.
- Write once.
- Passing `allclose` at tight tolerance by end of day is the realistic target.

**Write the `gradcheck` test today, before the backward.** float64, against the PyTorch reference. Having the oracle in place first is the difference between a 2-day backward and a 4-day one.

---

### Day 12 (Aug 23) — REST

Non-negotiable. Six consecutive working days by this point.

---

### Day 13–14 (Aug 24–25) — Fused RMSNorm, backward

The first genuinely hard thing.

- Input gradient: straightforward.
- **Weight gradient: the problem.** Every row contributes to the same weight vector. Requires atomics or a two-stage reduction with a partial-sum buffer. Getting it wrong produces subtly incorrect gradients that still look plausible.
- Verify with `torch.autograd.gradcheck` in float64.

> **Hard checkpoint:** if `gradcheck` is not passing by end of Day 14, ship the forward kernel, document the backward as in-progress, and move on. A shipped partial result beats a perfect one you are still debugging in September.

---

### Day 15 (Aug 26) — Integration, profiling, first benchmarks

- Wire both kernels into the model behind a flag: `RMSNorm(..., use_triton=False)`. Reference implementation stays in the repo.
- `torch.profiler` on a full training step. **Record what fraction of step time each op occupies.** This number is the backbone of the whole write-up.
- Benchmark: isolated op speedup, sweeping sequence length (512 → 8k) and hidden dim.
- Baseline is **`torch.compile`'d PyTorch, not eager.** Eager is a strawman and any reviewer will know it.

---

### Day 16 (Aug 27) — REST

---

### Day 17 (Aug 28) — Fused SwiGLU _(conditional)_

Only if the Day 15 profile shows SwiGLU is a meaningful share of step time. If it is 2%, **skip it and say so in the README.** The restraint is part of the signal.

- Fuse `Swish(xW) * (xV)` elementwise portion, saving an activation-sized intermediate.
- Backward needs the Swish derivative.

---

### Day 18–19 (Aug 29–30) — Chunked cross-entropy

**The highest-value kernel in the project.**

The logits tensor is `(B, T, vocab_size)` — at tiktoken's ~50k vocab this is typically the single largest allocation in the model, larger than any activation. PyTorch materializes logits, then softmax, then computes NLL.

- Compute in chunks over the vocabulary dimension, accumulating, without ever holding full logits in HBM.
- Backward is more tractable than RMSNorm's — the softmax gradient has a clean closed form.
- **The result you report is memory, not speed:** peak VRAM reduction, and max batch size before OOM with the kernel on vs. off.

---

### Day 20 (Aug 31) — Stacking experiment + full benchmark suite

The most sophisticated result in the repo, and the one nobody else has.

- Benchmark each kernel individually, then all together.
- **Show that the end-to-end gain is less than the sum of the parts.** It always is: as the memory-bound ops get faster, the matmuls become a larger share and Amdahl bites harder.
- Re-profile and show the new distribution of step time next to the old one.

---

## Phase 3 — Packaging & Launch (Days 21–24, Sep 1–4)

### Day 21 (Sep 1) — Refactor & test

- Structure into a package: `/src/model`, `/src/kernels`, `/src/training`, `/src/utils`.
- Type annotations, docstrings, `pytest`.
- **Include a unit test asserting the Newton-Schulz-style correctness properties of each kernel** — e.g. `gradcheck` passing, `allclose` against reference at tight tolerance. These tests signal more care than a hundred docstrings.

### Day 22 (Sep 2) — Plots

1. **Profile before/after** — step time breakdown by op, stacked bar. The headline figure.
2. **Isolated op speedup** vs. sequence length and hidden dim, with the crossover point where PyTorch wins (if there is one — show it).
3. **VRAM / max batch size** — cross-entropy kernel on vs. off.
4. **KV-cache latency** vs. context length (1k → 8k), from Day 6.
5. **MFU** during training.
6. Error bars across ≥3 seeds anywhere a training curve appears.

### Day 23 (Sep 3) — README

Structure it as the actual engineering narrative, not a feature checklist:

1. **Architecture** — RoPE, GQA, SwiGLU, RMSNorm, KV-cache. Diagram.
2. **Profile** — here is where time actually goes in this model.
3. **What I targeted and why** — the memory-bound ops, with the bandwidth-vs-compute reasoning.
4. **Per-kernel results** — including the ones that didn't help.
5. **Stacking + Amdahl** — why the total is less than the sum.
6. **How to run** — one-line install, one-line train.

**Reporting standard — this is what separates the repo from the median one:**

> Report three numbers together, always: _isolated op speedup_, _that op's share of step time_, _resulting end-to-end change._
> 
> "3.2× on the op, which is 7% of step time, yielding 4.8% end-to-end" reads as someone who understands Amdahl's law. "3.2× speedup!" reads as someone who doesn't know to check.

State plainly when a difference is within seed noise. Being the person who ran three seeds and said "this gap isn't distinguishable" is worth more than any additional feature.

### Day 24 (Sep 4) — Publish & outreach prep

1. Repository public.
2. Technical write-up: LinkedIn / X / personal blog. Lead with the profile-driven framing, not the feature list.
3. CV: **Key Technical Projects**, top, direct link.
4. **Build the outreach list.** Named people at Mistral, Kyutai, FAIR Paris, INRIA, CMAP whose papers you have actually read. Polytechnique alumni first — they answer current students far more reliably than cold emails.

---

## Cut from the previous plan

|Item|Reason|
|---|---|
|MoE routing|Results are noise below a certain scale and token volume. Plotting them as findings reads worse than omitting them.|
|DPO loss|90 minutes, not a day. Meaningless without a model producing distinguishable completions.|
|DDP / FSDP|Untested code without multiple GPUs. Shipping boilerplate you never ran is negative signal.|
|Generic "paper reproduction sprint"|The kernels _are_ the reproduction, with original measurement attached.|
|Fused attention|`F.scaled_dot_product_attention` already dispatches to FlashAttention-2 / cuDNN. You will not beat it, and "my kernel is 3× slower" is worse than no kernel.|
|Anything matmul-shaped|Compute-bound. cuBLAS is at hardware peak. Nothing to reclaim.|

---

## Daily rhythm

```
09:00 – 10:00 | Theory & paper reading (derive on paper)
10:00 – 13:00 | High-intensity kernel block
13:00 – 14:00 | Lunch & reset
14:00 – 16:30 | Debug, verify, profile
16:30 – 17:00 | Commit + log notes. STOP THE POD.
```

**Rest days are Day 12 (Aug 23) and Day 16 (Aug 27).** They are in the plan because the plan surviving contact with reality matters more than the plan being maximally dense.

---

## Notes

**Triton disambiguation.** _OpenAI Triton_ (`triton-lang`, `pip install triton`) is the Python GPU kernel language used here — actively developed, latest release 3.7.1 (June 2026), and the default backend `torch.compile` lowers to. _NVIDIA Triton Inference Server_ is an unrelated model-serving product being wound down. Same name, nothing else in common.

**Reference implementations.** vLLM ships production Triton kernels for RoPE and RMSNorm — the exact two targeted here. Confirmation the targets are right. Read them **after** writing your own, as a comparison, not before.

**The larger variable.** The repo gets you past the capability filter. It does not create the opportunity. Outreach volume and specificity in Nov–Jan matters more than which kernel you wrote. Budget real time for it.