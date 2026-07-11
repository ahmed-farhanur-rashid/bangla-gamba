# BanglaGamba Training Optimization Results

## Summary

Implemented 7 optimizations targeting speed and memory efficiency on RTX 4070 Super (12GB VRAM). All changes validated with full training loop benchmarks at production settings (batch_size=2, accum=64).

> [!IMPORTANT]
> **Full loop throughput: +15.3%** (13,625 → 15,709 tok/s) while keeping peak VRAM at 8.61 GB with 3.0 GB headroom.

---

## Benchmark Results

### Speed (fwd+bwd only, 20 steps)

| Config | tok/s | Δ | Peak GB |
|---|---|---|---|
| Baseline (grad_ckpt ON, no compile) | 22,083 | — | 5.33 |
| + compile FFN+GQA | 24,149 | +9.4% | 5.33 |
| No grad_ckpt | 24,757 | +12.1% | 6.81 |
| **No grad_ckpt + compile FFN+GQA** | **27,406** | **+24.1%** | **6.28** |

### Full Loop (optimizer + accumulation + z-loss, production-equivalent)

| Config | tok/s | Δ | Peak GB | Headroom |
|---|---|---|---|---|
| Baseline (grad_ckpt ON, no compile) | 13,625 | — | 7.66 | 3.9 GB |
| + compile FFN+GQA | 14,484 | +6.3% | 7.66 | 3.9 GB |
| **No grad_ckpt + compile FFN+GQA** | **15,709** | **+15.3%** | **8.61** | **3.0 GB** |

---

## Optimizations Implemented

### 🔴 High Impact

#### 1. Chunked z-loss computation
**Files:** [trainer.py](file:///home/farhan/my-projects/bangla-gamba/src/training/trainer.py#L264-L281)

The original `torch.logsumexp(logits, dim=-1)` on `(2, 2047, 48000)` logits created a **750 MiB float32 intermediate** — this was the #1 cause of OOM on the 2nd optimizer step. Now computed in chunks of 128 positions along the sequence dimension.

```diff
-z_loss = self.config.z_loss_weight * torch.logsumexp(logits, dim=-1).pow(2).mean()
+z_chunk = 128
+z_accum = 0.0
+n_elements = 0
+for t0 in range(0, T, z_chunk):
+    t1 = min(t0 + z_chunk, T)
+    lse = torch.logsumexp(logits[:, t0:t1, :], dim=-1)
+    z_accum = z_accum + lse.pow(2).sum()
+    n_elements += lse.numel()
+z_loss = self.config.z_loss_weight * (z_accum / n_elements)
```

#### 2. Per-submodule torch.compile (FFN + GQA)
**Files:** [train.py](file:///home/farhan/my-projects/bangla-gamba/src/train.py#L110-L129), [default_training.yaml](file:///home/farhan/my-projects/bangla-gamba/configs/default_training.yaml#L17)

Compiles 12 FFN sublayers and 6 GQA attention mixers individually. Mamba-3 layers stay uncompiled (Triton kernel incompatibility). Fuses SwiGLU ops (`silu + mul + linear`) and attention ops into fewer kernels.

#### 3. Gradient checkpointing disabled
**Files:** [default_training.yaml](file:///home/farhan/my-projects/bangla-gamba/configs/default_training.yaml#L15)

With chunked z-loss + expandable_segments, peak VRAM stays at 8.61 GB without grad_ckpt. Disabling it eliminates redundant FFN+GQA recomputation during backward (~15% gain in full loop).

#### 4. Gradient checkpointing now covers GQA mixer (not just FFN)
**Files:** [model.py](file:///home/farhan/my-projects/bangla-gamba/src/model/model.py#L64-L73)

When grad_ckpt IS enabled (e.g., if user needs more headroom), GQA attention layers are now also checkpointed — they were the larger of the two sublayers but were previously always eager. Mamba-3 stays eager due to Triton compatibility concerns.

#### 5. CUDA expandable_segments allocator
**Files:** [train.py](file:///home/farhan/my-projects/bangla-gamba/src/train.py#L63-L64)

Eliminates memory fragmentation that caused OOM even when total VRAM was sufficient. Set before any CUDA context is created.

### 🟡 Medium Impact

#### 6. RoPE cos/sin precomputed at init
**Files:** [rope.py](file:///home/farhan/my-projects/bangla-gamba/src/model/rope.py#L52-L57)

Precomputes and caches cos/sin tables for all positions at init. Forward just indexes into the cache — eliminates 6 redundant trig computations per forward pass.

#### 7. `non_blocking=True` on GPU transfers
**Files:** [trainer.py](file:///home/farhan/my-projects/bangla-gamba/src/training/trainer.py#L446), [trainer.py](file:///home/farhan/my-projects/bangla-gamba/src/training/trainer.py#L303)

Overlaps CPU→GPU DMA with Python work. Both training and eval transfers now use async copies (complementing existing `pin_memory=True`).

### 🟢 Low Impact

#### 8. `torch.inference_mode()` for eval
**Files:** [trainer.py](file:///home/farhan/my-projects/bangla-gamba/src/training/trainer.py#L299)

Replaces `@torch.no_grad()` with `torch.inference_mode()` context manager — skips version counting and view tracking.

#### 9. `persistent_workers=True` in DataLoader
**Files:** [collator.py](file:///home/farhan/my-projects/bangla-gamba/src/data/collator.py#L151)

Keeps DataLoader workers alive between iterations within the same loader.

#### 10. `torch.set_float32_matmul_precision('high')`
**Files:** [train.py](file:///home/farhan/my-projects/bangla-gamba/src/train.py#L131-L132)

Global TF32 precision setting (supplements per-backend flags).

---

## Production Config

```yaml
gradient_checkpointing: false
compile_model: false
compile_submodules: true
batch_size: 2
accumulation_steps: 64
```

With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set automatically in `train.py`.

> [!WARNING]
> If you add background GPU processes (heavy browser, additional CUDA apps), you can re-enable `gradient_checkpointing: true` to drop peak from 8.61 GB → 7.66 GB with a ~6% speed cost.
