"""
Evaluation Helper Functions for BanglaGamba Generation & Perplexity Benchmarking.

Provides clean evaluation pipelines, perplexity computations, multi-domain prompt testing,
and figure plotting functions for notebooks without code or warning clutter.
"""

from __future__ import annotations

import time
import math
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120


# ── 1. Model & Environment Auditing ──────────────────────────────────────────

def audit_environment_and_model(model, tokenizer) -> pd.DataFrame:
    """Audit model structure, parameter counts, and environment details."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    records = [
        {"Property": "Hugging Face Model ID", "Value": "ahmed-farhanur-rashid/bangla-gamba"},
        {"Property": "Model Class", "Value": type(model).__name__},
        {"Property": "Tokenizer Class", "Value": type(tokenizer).__name__},
        {"Property": "Total Parameters", "Value": f"{total_params / 1e6:.1f}M ({total_params:,})"},
        {"Property": "Trainable Parameters", "Value": f"{trainable_params / 1e6:.1f}M ({trainable_params:,})"},
        {"Property": "Vocabulary Size", "Value": f"{tokenizer.vocab_size:,}"},
        {"Property": "Context Window (Seq Len)", "Value": f"{getattr(model.config, 'seq_len', 2048)} tokens"},
        {"Property": "Model Device", "Value": str(next(model.parameters()).device)},
        {"Property": "Model Data Type", "Value": str(next(model.parameters()).dtype)},
    ]
    return pd.DataFrame(records)


# ── 2. Multi-Domain Generation Benchmark ────────────────────────────────────

def evaluate_generation_suite(model, tokenizer, prompts: List[Dict[str, str]], **gen_kwargs) -> pd.DataFrame:
    """Execute text generation across a suite of prompts and benchmark latency."""
    results = []

    for item in prompts:
        category = item.get("category", "General")
        prompt = item.get("prompt", "")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time

        gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tps = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0.0

        full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        results.append({
            "Category": category,
            "Prompt": prompt,
            "Generated Extension": gen_decoded.strip(),
            "Full Output": full_decoded.strip(),
            "New Tokens": gen_tokens,
            "Latency (s)": round(elapsed_sec, 3),
            "Speed (tokens/s)": round(tps, 1)
        })

    return pd.DataFrame(results)


# ── 3. Perplexity & Cross-Entropy Evaluation ─────────────────────────────────

def evaluate_perplexity_suite(model, tokenizer, test_sentences: List[Dict[str, str]]) -> pd.DataFrame:
    """Compute exact cross-entropy loss and perplexity (PPL) on test sentences."""
    results = []

    for item in test_sentences:
        domain = item.get("domain", "General")
        text = item.get("text", "")

        encodings = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            ppl = math.exp(loss) if loss < 20 else float("inf")

        results.append({
            "Domain": domain,
            "Text": text[:60] + "..." if len(text) > 60 else text,
            "Sequence Length": input_ids.shape[1],
            "Cross-Entropy Loss": round(loss, 4),
            "Perplexity (PPL)": round(ppl, 2)
        })

    return pd.DataFrame(results)


# ── 4. Logit Stability & Trajectory Audit ────────────────────────────────────

def audit_logit_trajectory(model, tokenizer, prompt: str = "বাংলাদেশের রাজধানী", steps: int = 40) -> pd.DataFrame:
    """Audit min/max logit trajectory and check for NaN/Inf across decoding steps."""
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    curr_ids = inputs.clone()
    records = []

    for step in range(steps):
        with torch.no_grad():
            outputs = model(curr_ids)
            last_logits = outputs.logits[:, -1, :].float()

            has_nan = torch.isnan(last_logits).any().item()
            has_inf = torch.isinf(last_logits).any().item()
            min_val = last_logits.min().item()
            max_val = last_logits.max().item()

            records.append({
                "Step": step,
                "SeqLen": curr_ids.shape[1],
                "Min Logit": round(min_val, 2),
                "Max Logit": round(max_val, 2),
                "NaN Detected": has_nan,
                "Inf Detected": has_inf,
            })

            if has_nan or has_inf:
                break

            next_tok = torch.argmax(last_logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)

    return pd.DataFrame(records)


# ── 5. Figure Plotting Functions ─────────────────────────────────────────────

def plot_perplexity_comparison(df_ppl: pd.DataFrame):
    """Plot Cross-Entropy Loss and Perplexity across evaluation domains."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    sns.barplot(data=df_ppl, x="Domain", y="Cross-Entropy Loss", hue="Domain", legend=False, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Cross-Entropy Loss Across Test Domains")
    axes[0].set_ylabel("Cross-Entropy Loss (Nats)")
    axes[0].set_xlabel("")

    for p in axes[0].patches:
        h = p.get_height()
        axes[0].annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    sns.barplot(data=df_ppl, x="Domain", y="Perplexity (PPL)", hue="Domain", legend=False, ax=axes[1], palette="Greens_d")
    axes[1].set_title("Language Model Perplexity (PPL = exp(Loss))")
    axes[1].set_ylabel("Perplexity (Lower is Better)")
    axes[1].set_xlabel("")

    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_logit_trajectory(df_traj: pd.DataFrame):
    """Plot min/max logit trajectory across autoregressive decoding steps."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(df_traj["Step"], df_traj["Max Logit"], marker="o", color="#2b5c8f", label="Max Logit", linewidth=2)
    ax.plot(df_traj["Step"], df_traj["Min Logit"], marker="s", color="#d95f02", label="Min Logit", linewidth=2)
    ax.axhline(30.0, color="red", linestyle="--", alpha=0.6, label="Clamping Upper Bound (+30.0)")
    ax.axhline(-30.0, color="red", linestyle="--", alpha=0.6, label="Clamping Lower Bound (-30.0)")

    ax.set_title("Autoregressive Logit Bounds Across 40 Decoding Steps")
    ax.set_xlabel("Decoding Step")
    ax.set_ylabel("Logit Value")
    ax.legend(loc="right")

    plt.tight_layout()
    plt.show()


def plot_decoding_latency_comparison(df_greedy: pd.DataFrame, df_sample: pd.DataFrame):
    """Plot token generation speed comparison between Greedy and Sampling decoding."""
    df_combined = pd.concat([
        df_greedy.assign(Strategy="Greedy (do_sample=False)"),
        df_sample.assign(Strategy="Sampling (temp=0.7, top_p=0.9)")
    ])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.barplot(data=df_combined, x="Category", y="Speed (tokens/s)", hue="Strategy", ax=ax, palette="Set2")

    ax.set_title("Generation Speed (Tokens/Second) Across Prompts and Decoding Strategies")
    ax.set_ylabel("Tokens / Second")
    ax.set_xlabel("")
    ax.legend(title="Decoding Strategy")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()
