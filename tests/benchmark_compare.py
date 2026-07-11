"""
Comprehensive speed + memory benchmark for all optimization configurations.

Runs multiple config combinations and outputs a comparison table.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python tests/benchmark_compare.py
"""

import sys
import os
import time
import gc

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.model.config import BanglaGambaConfig
from src.model.model import BanglaGambaModel
from src.model.optim import build_optimizers, load_optimizer_config, build_param_groups
from src.training.scheduler import build_schedulers
from src.utils.seed import set_seed


def run_config(
    label: str,
    grad_ckpt: bool,
    compile_ffn: bool,
    compile_gqa: bool,
    batch_size: int = 2,
    steps: int = 20,
    warmup: int = 5,
    full_loop: bool = False,
    accum: int = 4,
    opt_steps: int = 2,
):
    """Run a single configuration and return results."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_seed(1839)

    config = BanglaGambaConfig.from_yaml("configs/banglagamba_12l.yaml")
    model = BanglaGambaModel(config).to("cuda")

    if grad_ckpt:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    if compile_ffn:
        for layer in model.layers:
            layer.ffn = torch.compile(layer.ffn)
    if compile_gqa:
        for layer in model.layers:
            if layer.layer_type == "attn":
                layer.mixer = torch.compile(layer.mixer)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    seq_len = 2048
    dummy_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

    if full_loop:
        # Full training loop with optimizer + accumulation
        opt_config = load_optimizer_config("configs/muon_adamw.yaml")
        muon_opt, adamw_opt = build_optimizers(model, opt_config)
        muon_params, adamw_params = build_param_groups(model)
        muon_sched, adamw_sched = build_schedulers(
            muon_opt, adamw_opt, warmup_steps=15, total_steps=1000, min_lr_ratio=0.1,
        )
        model.train()
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)

        # Warmup (includes compile)
        for i in range(warmup):
            inp = dummy_ids[:, :-1]
            tgt = dummy_ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), tgt.reshape(-1))
                # Chunked z-loss
                B, T, V = logits.shape
                z_chunk = 128
                z_accum_val = 0.0
                n_el = 0
                for t0 in range(0, T, z_chunk):
                    t1 = min(t0 + z_chunk, T)
                    lse = torch.logsumexp(logits[:, t0:t1, :], dim=-1)
                    z_accum_val = z_accum_val + lse.pow(2).sum()
                    n_el += lse.numel()
                z_loss = 1e-4 * (z_accum_val / n_el)
                total = (loss + z_loss) / accum
            total.backward()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed full loop
        t0 = time.perf_counter()
        micro = 0
        for opt_step in range(opt_steps):
            for _ in range(accum):
                inp = dummy_ids[:, :-1]
                tgt = dummy_ids[:, 1:].contiguous()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(inp)
                    B, T, V = logits.shape
                    ce = F.cross_entropy(logits.view(-1, V), tgt.view(-1))
                    z_chunk = 128
                    z_a = 0.0
                    n_e = 0
                    for t0_z in range(0, T, z_chunk):
                        t1_z = min(t0_z + z_chunk, T)
                        lse = torch.logsumexp(logits[:, t0_z:t1_z, :], dim=-1)
                        z_a = z_a + lse.pow(2).sum()
                        n_e += lse.numel()
                    z_l = 1e-4 * (z_a / n_e)
                    total = (ce + z_l) / accum
                total.backward()
                micro += 1

            torch.nn.utils.clip_grad_norm_(
                list(muon_params) + list(adamw_params), max_norm=1.0
            )
            muon_opt.step()
            adamw_opt.step()
            muon_sched.step()
            adamw_sched.step()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        total_tokens = batch_size * (seq_len - 1) * micro
    else:
        # Speed-only benchmark (no optimizer)
        model.train()

        # Warmup
        for i in range(warmup):
            inp = dummy_ids[:, :-1]
            tgt = dummy_ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), tgt.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed benchmark
        t0 = time.perf_counter()
        for _ in range(steps):
            inp = dummy_ids[:, :-1]
            tgt = dummy_ids[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), tgt.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        total_tokens = batch_size * (seq_len - 1) * steps

    tok_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / (steps if not full_loop else micro)) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "tok_per_sec": tok_per_sec,
        "ms_per_step": ms_per_step,
        "peak_mem_gb": peak_mem,
    }


def main():
    print("=" * 72)
    print("  BanglaGamba Optimization Comparison Benchmark")
    print("  expandable_segments: True")
    print("=" * 72)

    configs = [
        ("Baseline (grad_ckpt ON)",           True,  False, False),
        ("+ compile FFN+GQA",                 True,  True,  True),
        ("No grad_ckpt",                      False, False, False),
        ("No grad_ckpt + compile FFN+GQA",    False, True,  True),
    ]

    # --- Speed benchmarks (fwd+bwd only) ---
    print("\n" + "─" * 72)
    print("  Speed Benchmark (20 fwd+bwd steps, no optimizer)")
    print("─" * 72)
    speed_results = []
    for label, gc_flag, cffn, cgqa in configs:
        print(f"\n  → {label}...")
        r = run_config(label, gc_flag, cffn, cgqa, steps=20, warmup=5)
        speed_results.append(r)
        print(f"    tok/s={r['tok_per_sec']:,.0f}  ms/step={r['ms_per_step']:.1f}  peak={r['peak_mem_gb']:.2f}GB")

    baseline = speed_results[0]["tok_per_sec"]
    print(f"\n  {'Config':<40s} {'tok/s':>8s} {'Δ':>8s} {'ms/step':>8s} {'Peak GB':>8s}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for r in speed_results:
        delta = ((r["tok_per_sec"] / baseline) - 1) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {r['label']:<40s} {r['tok_per_sec']:>8,.0f} {sign}{delta:>6.1f}% {r['ms_per_step']:>7.1f} {r['peak_mem_gb']:>7.2f}")

    # --- Full loop benchmarks ---
    print("\n" + "─" * 72)
    print("  Full Loop Benchmark (2 opt steps × 4 micro-batches = 8 fwd+bwd)")
    print("─" * 72)
    full_configs = [
        ("Baseline (grad_ckpt ON)",           True,  False, False),
        ("+ compile FFN+GQA",                 True,  True,  True),
        ("No grad_ckpt + compile FFN+GQA",    False, True,  True),
    ]
    full_results = []
    for label, gc_flag, cffn, cgqa in full_configs:
        print(f"\n  → {label}...")
        try:
            r = run_config(
                label, gc_flag, cffn, cgqa,
                full_loop=True, accum=4, opt_steps=2, warmup=3,
            )
            full_results.append(r)
            print(f"    tok/s={r['tok_per_sec']:,.0f}  ms/step={r['ms_per_step']:.1f}  peak={r['peak_mem_gb']:.2f}GB")
        except torch.OutOfMemoryError:
            print(f"    ❌ OOM!")
            full_results.append({"label": label, "tok_per_sec": 0, "ms_per_step": 0, "peak_mem_gb": 0})
            gc.collect()
            torch.cuda.empty_cache()

    if full_results:
        fb = full_results[0]["tok_per_sec"]
        print(f"\n  {'Config':<40s} {'tok/s':>8s} {'Δ':>8s} {'ms/step':>8s} {'Peak GB':>8s}")
        print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
        for r in full_results:
            if r["tok_per_sec"] > 0 and fb > 0:
                delta = ((r["tok_per_sec"] / fb) - 1) * 100
                sign = "+" if delta >= 0 else ""
                print(f"  {r['label']:<40s} {r['tok_per_sec']:>8,.0f} {sign}{delta:>6.1f}% {r['ms_per_step']:>7.1f} {r['peak_mem_gb']:>7.2f}")
            else:
                print(f"  {r['label']:<40s} {'OOM':>8s} {'':>8s} {'':>8s} {'':>8s}")

    print(f"\n{'=' * 72}")
    print("  VRAM budget: 12.0 GB (RTX 4070 Super)")
    print("  System overhead: ~0.5 GB (desktop compositor, browser)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
