"""
Speed benchmark for BanglaGamba forward+backward passes.

Measures tok/s throughput with synthetic data to isolate model speed
from data pipeline. Run this BEFORE and AFTER each optimization.

Usage:
    python tests/benchmark_speed.py [--batch-size N] [--steps N] [--grad-ckpt] [--compile-ffn]
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model.config import BanglaGambaConfig
from src.model.model import BanglaGambaModel
from src.utils.seed import set_seed


def benchmark(
    batch_size: int = 2,
    seq_len: int = 2048,
    steps: int = 20,
    warmup: int = 5,
    grad_ckpt: bool = True,
    compile_ffn: bool = False,
    compile_gqa: bool = False,
):
    set_seed(1839)
    device = "cuda"

    # Build model
    config = BanglaGambaConfig.from_yaml("configs/banglagamba_12l.yaml")
    model = BanglaGambaModel(config).to(device)

    if grad_ckpt:
        model.gradient_checkpointing_enable()
        print("[Bench] Gradient checkpointing: ON")
    else:
        model.gradient_checkpointing_disable()
        print("[Bench] Gradient checkpointing: OFF")

    if compile_ffn:
        for layer in model.layers:
            layer.ffn = torch.compile(layer.ffn)
        print("[Bench] torch.compile(FFN): ON")

    if compile_gqa:
        for layer in model.layers:
            if layer.layer_type == "attn":
                layer.mixer = torch.compile(layer.mixer)
        print("[Bench] torch.compile(GQA): ON")

    model.train()

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Synthetic data
    dummy_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup (includes torch.compile tracing if enabled)
    print(f"\n[Bench] Warming up ({warmup} steps)...")
    for i in range(warmup):
        input_ids = dummy_ids[:, :-1]
        targets = dummy_ids[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        if compile_ffn or compile_gqa:
            print(f"  warmup step {i+1}/{warmup} done")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"\n[Bench] Running {steps} steps (batch_size={batch_size}, seq_len={seq_len})...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(steps):
        input_ids = dummy_ids[:, :-1]
        targets = dummy_ids[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tokens_per_step = batch_size * (seq_len - 1)
    total_tokens = tokens_per_step * steps
    tok_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / steps) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n{'='*60}")
    print(f"  BanglaGamba Speed Benchmark Results")
    print(f"{'='*60}")
    print(f"  batch_size:      {batch_size}")
    print(f"  seq_len:         {seq_len}")
    print(f"  grad_ckpt:       {grad_ckpt}")
    print(f"  compile_ffn:     {compile_ffn}")
    print(f"  compile_gqa:     {compile_gqa}")
    print(f"{'='*60}")
    print(f"  Steps:           {steps}")
    print(f"  Total time:      {elapsed:.2f}s")
    print(f"  ms/step:         {ms_per_step:.1f}")
    print(f"  tok/s:           {tok_per_sec:,.0f}")
    print(f"  Peak VRAM:       {peak_mem:.2f} GB")
    print(f"{'='*60}")

    return {
        "tok_per_sec": tok_per_sec,
        "ms_per_step": ms_per_step,
        "peak_mem_gb": peak_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="BanglaGamba Speed Benchmark")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing (default: off)")
    parser.add_argument("--compile-ffn", action="store_true",
                        help="torch.compile each FFN sublayer")
    parser.add_argument("--compile-gqa", action="store_true",
                        help="torch.compile each GQA sublayer")
    args = parser.parse_args()

    benchmark(
        batch_size=args.batch_size,
        seq_len=2048,
        steps=args.steps,
        warmup=args.warmup,
        grad_ckpt=args.grad_ckpt,
        compile_ffn=args.compile_ffn,
        compile_gqa=args.compile_gqa,
    )


if __name__ == "__main__":
    main()
