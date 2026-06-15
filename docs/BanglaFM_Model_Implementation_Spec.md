# BanglaFM-12L: Model Implementation Specification

## Implementation handoff document — architecture, optimizer, and training loop only

This document specifies a single, final configuration for implementation. It supersedes the architecture (§1) and optimizer (§4.4) sections of the previous guide. Tokenizer training, dataset sourcing, mixing ratios, and paper-writing content are **out of scope** here — this is purely "build the model and training loop."

---

## 0. TARGET SUMMARY

| Property | Value |
|---|---|
| Architecture | Mamba-2 / GQA hybrid, 1:1 interleaved |
| Total blocks | 12 (6 Mamba-2 + 6 GQA, each paired with SwiGLU FFN) |
| Hidden size (d_model) | 1024 |
| Total parameters | ~199M |
| Vocabulary size | 48,000 |
| Sequence length | 2048 (fixed, no curriculum) |
| Token budget | ~5B tokens (single run, one epoch) |
| Target | Beat BanglaBERT-base (SentNoB macro-F1 72.89) |
| Hardware | RTX 4070 Super, 12GB VRAM |
| Optimizer | Hybrid Muon (2D matmul weights) + AdamW (everything else) |

---

## 1. ARCHITECTURE

### 1.1 Layer Pattern

12 blocks total, strict 1:1 alternation, starting with Mamba-2 and ending with GQA — Mamba-2 handles early local/positional mixing cheaply; the final GQA layer gives the model one global "lookup" pass before the LM head.

```python
# 0 = Mamba-2 block, 1 = GQA block
layer_types = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
```

Each block = `[Mixer (Mamba-2 OR GQA)] + [SwiGLU FFN]`, both pre-normed with RMSNorm, both with residual connections. This is the standard Jamba/Zamba block shape — do not omit the FFN on Mamba-2 blocks.

### 1.2 Full Hyperparameter Table

| Hyperparameter | Value | Notes |
|---|---|---|
| `hidden_size` (d_model) | 1024 | |
| `num_hidden_layers` | 12 | |
| `vocab_size` | 48000 | divisible by 128 |
| `max_position_embeddings` | 2048 | fixed throughout training |
| `tie_word_embeddings` | true | saves ~49M params at this scale |
| **Mamba-2 mixer** | | |
| `mamba_d_state` | 128 | SSD state dim (Mamba-2, not Mamba-1's d_state=16) |
| `mamba_d_conv` | 4 | local conv width |
| `mamba_expand` | 2 | d_inner = 2048 |
| `mamba_headdim` | 64 | → 32 internal SSM heads |
| `mamba_ngroups` | 1 | default |
| `mamba_rmsnorm` | true | Mamba-2's internal gated RMSNorm (keep on) |
| `mamba_chunk_size` | 256 | SSD chunked-scan size, default is fine at 2048 ctx |
| **GQA mixer** | | |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 4 | 4:1 GQA ratio |
| `head_dim` | 64 | 16×64 = 1024 |
| `rope_theta` | 10000.0 | |
| `qk_norm` | true | **new vs. previous guide — see §6.1** |
| **FFN (SwiGLU)** | | |
| `intermediate_size` | 2560 | `⌊(2/3 × 4 × 1024)/256⌋×256 = 2560` |
| `hidden_act` | silu | |
| **Norm** | | |
| `rms_norm_eps` | 1e-5 | |
| `norm_type` | RMSNorm, pre-norm only | one final RMSNorm before LM head |

### 1.3 Sub-component Implementation Notes

**Mamba-2 mixer** — use `mamba_ssm.Mamba2` directly:
```python
from mamba_ssm import Mamba2
mixer = Mamba2(
    d_model=1024, d_state=128, d_conv=4, expand=2,
    headdim=64, ngroups=1, rmsnorm=True, chunk_size=256,
)
```
Do **not** apply RoPE or any positional encoding inside Mamba-2 blocks — the SSM recurrence encodes position implicitly.

**GQA mixer** — RoPE applied to Q/K only, inside GQA blocks exclusively. Use Flash Attention 2 for these layers.
```python
import flash_attn
# q: [B, T, n_heads, head_dim], k/v: [B, T, n_kv_heads, head_dim]
# repeat k,v heads to n_heads // n_kv_heads before/inside the FA2 call
# (flash_attn_func supports GQA natively via differing q/k head counts in recent versions —
#  verify version; otherwise expand kv heads manually)
```

**SwiGLU FFN:**
```python
def swiglu_ffn(x, W_gate, W_up, W_down):
    return (F.silu(x @ W_gate) * (x @ W_up)) @ W_down
```

**RMSNorm:** pre-norm before every mixer and every FFN (i.e., 2 RMSNorms per block — `norm1` before mixer, `norm2` before FFN), plus one final RMSNorm before the LM head. No LayerNorm anywhere, no post-norm.

---

## 2. EMBEDDING / VOCABULARY INTEGRATION

- `vocab_size = 48000`, `tie_word_embeddings = true` (input embedding matrix == LM head weight, transposed).
- Embedding table shape: `[48000, 1024]` ≈ 49.15M params (tied — counted once).
- Special tokens occupy IDs 0–145 (146 reserved IDs: pad/unk/bos/eos, ChatML roles, task-control tokens, language tags, sentiment labels, reasoning/CoT markers, document-structure markers, 100 reserved slots). Normal subword vocab occupies IDs 146–47999 (47,854 slots).
- **Initialization:** embedding weights init with `std = 1/sqrt(hidden_size)` (≈0.03125), not the default `0.02` — this keeps embedding-scale activations consistent with RMSNorm's expected input scale at d_model=1024. Apply the same std to the (tied) LM head.
- Sentiment/task label tokens (`<|positive|>`, `<|negative|>`, `<|neutral|>`, `<|mixed|>`, `<|offensive|>`, etc.) get learned embeddings from step 1 — no post-hoc resizing needed.

---

## 3. LOSS FUNCTION

Standard causal LM cross-entropy, **plus a z-loss term** (see §6.2):

```python
logits = model(input_ids)                          # [B, T, V]
ce_loss = F.cross_entropy(
    logits.view(-1, vocab_size), targets.view(-1), ignore_index=PAD_ID
)
z_loss = 1e-4 * torch.logsumexp(logits, dim=-1).pow(2).mean()
loss = ce_loss + z_loss
```

---

## 4. OPTIMIZER: HYBRID MUON + ADAMW

This is the core change from the previous guide (which used Fused AdamW for everything).

### 4.1 Param Group Assignment

**Muon group** — all large 2D dense matmul weights:
- Mamba-2: `in_proj.weight`, `out_proj.weight`
- GQA: `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`
- FFN: `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`

**AdamW group** — everything else:
- Embedding / LM head (tied): `embed_tokens.weight`
- All RMSNorm weights (block pre-norms, final norm, Mamba-2's internal gated norm)
- Mamba-2 1D/scalar params: `conv1d.weight`, `conv1d.bias`, `A_log`, `D`, `dt_bias`
- Any biases (there shouldn't be many — `mamba_bias=false`, FFN/attention projections have no bias)

```python
def build_param_groups(model):
    muon_params, adamw_params = [], []
    muon_name_substrings = (
        "in_proj.weight", "out_proj.weight",          # Mamba-2
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",  # GQA
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",           # FFN
    )
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(s in name for s in muon_name_substrings) and p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
```

**Sanity check before training:** log the total param count in each group. Expect roughly: Muon ≈ 150M, AdamW ≈ 49M (dominated by the tied embedding). If the embedding accidentally lands in the Muon group (it's 2D — `[48000, 1024]`), the name-substring filter above will correctly exclude it since `"embed_tokens.weight"` doesn't match any Muon substring. Verify explicitly with an assertion, not just by inspection.

### 4.2 Optimizer Hyperparameters

```python
muon_optimizer = Muon(
    muon_params,
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,            # Newton-Schulz orthogonalization iterations
    weight_decay=0.01,
)

adamw_optimizer = torch.optim.AdamW(
    adamw_params,
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,
)
```

### 4.3 Muon Implementation Notes

Use a standard reference Muon implementation (Newton-Schulz orthogonalized momentum SGD). Key points for correctness:

1. **Orthogonalization step**: each 2D weight's momentum buffer is passed through a quintic Newton-Schulz iteration to approximate `U @ V^T` from its SVD `U @ S @ V^T` (i.e., zero out singular values, keep singular vectors). Use the standard tuned 5-step coefficient set from a reference Muon implementation — do not hand-derive new coefficients.
2. **Precision**: run the Newton-Schulz iteration in bf16 for speed, but the momentum buffer itself should accumulate in fp32 to avoid drift over thousands of steps.
3. **Non-square matrices**: for `in_proj`/`gate_proj`/`up_proj` (which expand dimensions, e.g., 1024→2048 or 1024→2560), Newton-Schulz still applies — it operates on the matrix regardless of aspect ratio, but reference implementations typically transpose so the iteration runs on the smaller dimension for efficiency. Confirm the implementation handles non-square shapes (all of this model's Muon matrices are non-square except none — double check `o_proj`/`down_proj` which map back to 1024).
4. **Distributed/single-GPU**: single-GPU here, so skip any all-reduce logic present in multi-GPU Muon implementations.

### 4.4 Learning Rate Schedules

Both optimizers use the **same step-based schedule shape** (linear warmup → cosine decay to 10% of peak) but **different peak LRs** (0.02 for Muon, 3e-4 for AdamW). Warmup = 1.5% of total steps.

```python
import math

def get_lr_multiplier(step, warmup_steps, total_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

# total_steps computed from token budget / effective_batch_tokens (see §5.2)
muon_sched  = torch.optim.lr_scheduler.LambdaLR(muon_optimizer,  lambda s: get_lr_multiplier(s, warmup_steps, total_steps))
adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_optimizer, lambda s: get_lr_multiplier(s, warmup_steps, total_steps))
```

### 4.5 Gradient Clipping

Clip across **both** param groups together using a single global norm (standard global clipping, applied before either optimizer's `.step()`):

```python
torch.nn.utils.clip_grad_norm_(
    list(muon_params) + list(adamw_params), max_norm=1.0
)
```

### 4.6 Step Order

```python
muon_optimizer.step()
adamw_optimizer.step()
muon_sched.step()
adamw_sched.step()
muon_optimizer.zero_grad(set_to_none=True)
adamw_optimizer.zero_grad(set_to_none=True)
```

### 4.7 First-300-Steps Monitoring (Critical for a Single Run)

Log gradient norms **separately** for the Muon and AdamW param groups for at least the first 300 steps. If the Muon group's gradient norm is wildly different in scale from the AdamW group's (more than ~2 orders of magnitude), the Muon LR (0.02) likely needs adjustment before committing to the full run — this is the single most common cause of early instability when porting a Muon config to a new model size.

---

## 5. TRAINING LOOP

### 5.1 Mixed Precision

BF16 throughout, FP32 master weights maintained implicitly via optimizer state (Muon's fp32 momentum, AdamW's fp32 moments).

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = compute_loss(logits, targets)  # includes z-loss, §3
```

### 5.2 Gradient Accumulation

Physical batch size at 2048 ctx on 12GB: 2–4 sequences (this model is ~70% the size of the previous guide's Config B, so slightly larger physical batch is likely feasible — verify empirically). Target effective batch ≈ 256 sequences × 2048 tokens ≈ 524K tokens/step, or scale down to ≈131K tokens/step if memory-constrained — pick one and compute `total_steps = token_budget / effective_batch_tokens` for the LR schedule in §4.4.

```python
accumulation_steps = effective_batch_seqs // physical_batch_size

for i, batch in enumerate(dataloader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = compute_loss(model(batch["input_ids"]), batch["labels"]) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        muon_optimizer.step(); adamw_optimizer.step()
        muon_sched.step(); adamw_sched.step()
        muon_optimizer.zero_grad(set_to_none=True)
        adamw_optimizer.zero_grad(set_to_none=True)
```

### 5.3 Flash Attention 2

GQA layers only, as in §1.3. `pip install flash-attn --no-build-isolation`.

### 5.4 Gradient Checkpointing

At ~199M params (vs. the previous guide's 269M Config B), this is **optional rather than required** (see VRAM budget in §8) — but enable it anyway for headroom to push the physical batch size up, since larger physical batches reduce gradient-accumulation noise and slightly improve throughput:

```python
model.gradient_checkpointing_enable()
```

### 5.5 torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

Apply after model init and after wrapping with gradient checkpointing, before the training loop. Note: confirm `mamba_ssm`'s fused kernels and Muon's Newton-Schulz step are compile-compatible in your installed versions — if `torch.compile` errors on either, fall back to compiling only the FFN/attention submodules, or skip compilation for the optimizer step.

---

## 6. STABILITY OPTIMIZATIONS (ADDRESSING OVERSIGHTS)

These were not in the previous guide and are recommended additions given the switch to Muon and the single-shot constraint.

### 6.1 QK-Norm

Apply RMSNorm to Q and K projections (per-head) immediately after the GQA projection, before RoPE:

```python
q = rmsnorm_qk(q_proj(x))  # per-head RMSNorm, learnable scale
k = rmsnorm_qk(k_proj(x))
q, k = apply_rope(q, k)
```

Rationale: Muon's orthogonalized updates can change the effective scale of Q/K projections more aggressively step-to-step than AdamW would. QK-norm bounds attention logit magnitudes regardless of upstream weight scale drift, which is cheap insurance for a run you can't repeat. This is standard in several recent Muon-trained models.

### 6.2 Z-Loss

Included in §3. A small (`1e-4`) penalty on `logsumexp(logits)²` discourages the LM head from producing runaway logit magnitudes — a known failure mode in long single runs that otherwise manifests as a sudden loss spike late in training with no easy recovery. Negligible compute cost.

### 6.3 Residual-Branch Output Scaling at Init

Scale the initialization of `out_proj` (Mamba-2), `o_proj` (GQA), and `down_proj` (FFN) — i.e., the matrices writing back into the residual stream — by `1/sqrt(2 * num_hidden_layers)`:

```python
for name, p in model.named_parameters():
    if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
        p.data.mul_(1.0 / math.sqrt(2 * num_hidden_layers))
```

Rationale: standard GPT-2-style residual init scaling, prevents activation variance from growing with depth. With only 12 layers this is a smaller effect than at 16+, but it's a one-line addition with no downside.

### 6.4 Decoupled Weight Decay Consistency

Both `weight_decay` values above (Muon: 0.01, AdamW: 0.1) are *decoupled* weight decay (applied directly to weights, not folded into the gradient) — confirm whichever Muon implementation is used applies decoupled decay, matching AdamW's `fused=True` behavior. Mismatched decay semantics between the two groups is a subtle bug that won't crash training but will quietly bias the param-norm dynamics between groups.

---

## 7. CHECKPOINTING & REPRODUCIBILITY

```python
checkpoint = {
    "step": current_step,
    "tokens_seen": current_step * effective_batch_tokens,
    "val_perplexity": val_ppl,
    "val_ppl_bn": val_ppl_bangla,
    "val_ppl_en": val_ppl_english,
    "train_loss": loss.item(),
    "model_state_dict": model.state_dict(),
    "muon_optimizer_state_dict": muon_optimizer.state_dict(),
    "adamw_optimizer_state_dict": adamw_optimizer.state_dict(),
    "muon_sched_state_dict": muon_sched.state_dict(),
    "adamw_sched_state_dict": adamw_sched.state_dict(),
    "config": model_config,
    "rng_state": torch.get_rng_state(),
}
```

Save every 2,000–5,000 steps. Keep last 3 + best-by-validation-perplexity. Implement `--resume_from_checkpoint` (covering **both** optimizer states and **both** schedulers) before starting the full run — a resume path that restores the model but not the Muon momentum buffers will silently degrade training quality after every restart.

```python
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
random.seed(42); np.random.seed(42)
torch.backends.cudnn.deterministic = True
```

---

## 8. PARAMETER COUNT & VRAM BUDGET

| Component | Params |
|---|---|
| Embedding (tied) | ~49.15M |
| 6× Mamba-2 blocks (mixer + FFN) | ~86.8M |
| 6× GQA blocks (mixer + FFN) | ~62.9M |
| **Total** | **~199M** |
| — of which Muon group | ~150M |
| — of which AdamW group | ~49M |

| Memory item | Estimate |
|---|---|
| Model weights (bf16) | ~0.40 GB |
| Gradients (bf16) | ~0.40 GB |
| Muon momentum (fp32, 150M params) | ~0.60 GB |
| AdamW moments (fp32, 2×, 49M params) | ~0.39 GB |
| **Fixed total** | **~1.8 GB** |
| Remaining for activations (12GB card) | ~10 GB |

This model is comfortably smaller than the previous guide's Config B (269M) — the fixed memory footprint leaves substantial headroom for activations even without gradient checkpointing, though §5.4 still recommends enabling it to allow a larger physical batch size.
