"""
Realistic training loop simulation for BanglaGamba OOM testing.

Tests whether the real training loop fits in 12GB VRAM with all
components: model, optimizer state, gradient accumulation, z-loss.

Usage:
    python scripts/benchmark_full_loop.py [--optimizer-steps N] [--grad-ckpt]
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model.config import BanglaGambaConfig
from src.model.model import BanglaGambaModel
from src.model.optim import build_optimizers, load_optimizer_config, build_param_groups
from src.training.scheduler import build_schedulers
from src.utils.seed import set_seed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-steps", type=int, default=2,
                        help="Number of full optimizer steps (each = accum micro-batches)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=64)
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--compile-ffn", action="store_true", default=False)
    parser.add_argument("--compile-gqa", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(1839)
    device = "cuda"
    seq_len = 2048
    z_loss_weight = 1e-4
    max_grad_norm = 1.0

    # ── Build model ───────────────────────────────────────────────────
    config = BanglaGambaConfig.from_yaml("configs/banglagamba_12l.yaml")
    model = BanglaGambaModel(config).to(device)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        print("[Test] Gradient checkpointing: ON")
    else:
        model.gradient_checkpointing_disable()
        print("[Test] Gradient checkpointing: OFF")

    # ── Compile submodules ────────────────────────────────────────────
    if args.compile_ffn:
        for layer in model.layers:
            layer.ffn = torch.compile(layer.ffn)
        print("[Test] torch.compile(FFN): ON")
    if args.compile_gqa:
        for layer in model.layers:
            if layer.layer_type == "attn":
                layer.mixer = torch.compile(layer.mixer)
        print("[Test] torch.compile(GQA): ON")

    # ── Build optimizers (allocates optimizer state in VRAM) ──────────
    optimizer_config = load_optimizer_config("configs/muon_adamw.yaml")
    muon_optimizer, adamw_optimizer = build_optimizers(model, optimizer_config)

    # ── Build schedulers ─────────────────────────────────────────────
    muon_scheduler, adamw_scheduler = build_schedulers(
        muon_optimizer, adamw_optimizer,
        warmup_steps=15, total_steps=1000, min_lr_ratio=0.1,
    )

    # ── Param groups for clipping ─────────────────────────────────────
    muon_params, adamw_params = build_param_groups(model)

    # ── TF32 ──────────────────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model.train()
    muon_optimizer.zero_grad(set_to_none=True)
    adamw_optimizer.zero_grad(set_to_none=True)

    # Synthetic data
    dummy_ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    pad_token_id = 0

    # Show initial memory state (model + optimizer state, before any forward)
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"\n[Test] Base VRAM (model + optimizer state): {base_mem:.2f} GB")
    print(f"[Test] Available for activations: {11.59 - base_mem:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    total_micro = args.optimizer_steps * args.accum
    print(f"\n[Test] Running {args.optimizer_steps} optimizer steps "
          f"× {args.accum} micro-batches = {total_micro} fwd+bwd passes")
    print(f"[Test] batch_size={args.batch_size}, seq_len={seq_len}")

    # ── Warmup compile ────────────────────────────────────────────────
    if args.compile_ffn or args.compile_gqa:
        print("[Test] Triggering compilation...")
        input_ids = dummy_ids[:, :-1]
        targets = dummy_ids[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.reshape(-1))
        loss.backward()
        muon_optimizer.zero_grad(set_to_none=True)
        adamw_optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        print("[Test] Compilation done.\n")

    t0 = time.perf_counter()
    micro_step = 0

    for opt_step in range(1, args.optimizer_steps + 1):
        running_loss = 0.0

        for micro in range(args.accum):
            input_ids = dummy_ids[:, :-1]
            targets = dummy_ids[:, 1:].contiguous()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

                B, T, V = logits.shape
                ce_loss = F.cross_entropy(
                    logits.view(-1, V), targets.view(-1),
                    ignore_index=pad_token_id,
                )
                z_loss = z_loss_weight * torch.logsumexp(logits, dim=-1).pow(2).mean()
                total_loss = (ce_loss + z_loss) / args.accum

            total_loss.backward()

            running_loss += total_loss.item() * args.accum
            micro_step += 1

            if micro_step % 16 == 0:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                cur = torch.cuda.memory_allocated() / 1024**3
                print(f"  micro {micro_step:4d}/{total_micro} | "
                      f"cur={cur:.2f}GB peak={peak:.2f}GB")

        # ── Optimizer step ────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(
            list(muon_params) + list(adamw_params),
            max_norm=max_grad_norm,
        )
        muon_optimizer.step()
        adamw_optimizer.step()
        muon_scheduler.step()
        adamw_scheduler.step()
        muon_optimizer.zero_grad(set_to_none=True)
        adamw_optimizer.zero_grad(set_to_none=True)

        peak = torch.cuda.max_memory_allocated() / 1024**3
        cur = torch.cuda.memory_allocated() / 1024**3
        avg_loss = running_loss / args.accum

        print(f"\n  ✓ Optimizer step {opt_step}/{args.optimizer_steps} | "
              f"loss={avg_loss:.4f} | cur={cur:.2f}GB peak={peak:.2f}GB\n")

    t1 = time.perf_counter()
    elapsed = t1 - t0
    tokens_per_step = args.batch_size * (seq_len - 1)
    total_tokens = tokens_per_step * total_micro
    tok_per_sec = total_tokens / elapsed

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    cur_mem = torch.cuda.memory_allocated() / 1024**3

    print(f"{'='*60}")
    print(f"  Full Training Loop Simulation Results")
    print(f"{'='*60}")
    print(f"  grad_ckpt:        {args.grad_ckpt}")
    print(f"  compile_ffn:      {args.compile_ffn}")
    print(f"  compile_gqa:      {args.compile_gqa}")
    print(f"  Optimizer steps:  {args.optimizer_steps}")
    print(f"  Micro-batches:    {total_micro}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  accum:            {args.accum}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  tok/s:            {tok_per_sec:,.0f}")
    print(f"  Current VRAM:     {cur_mem:.2f} GB")
    print(f"  Peak VRAM:        {peak_mem:.2f} GB")
    print(f"  VRAM headroom:    {11.59 - peak_mem:.2f} GB")
    print(f"{'='*60}")

    if peak_mem > 11.0:
        print("\n  ⚠️  DANGER: Peak VRAM very close to 12GB limit!")
        print("     Training will likely OOM with any background GPU process.")
    elif peak_mem > 10.0:
        print("\n  ⚠️  WARNING: Peak VRAM >10GB. May OOM with browser/desktop GPU usage.")
    else:
        print(f"\n  ✅  SAFE: {11.59 - peak_mem:.1f} GB headroom. "
              f"Training should be stable.")


if __name__ == "__main__":
    main()
