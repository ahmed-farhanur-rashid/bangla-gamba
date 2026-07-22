"""
EDA Helper Functions for BanglaGamba Pretraining Corpus & Tokenizer Analysis.

Provides reusable data loading, structural auditing, statistical analysis,
and figure plotting functions for notebooks without print or warning clutter.
"""

from __future__ import annotations

import os
import glob
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress all seaborn/matplotlib warnings globally
warnings.filterwarnings("ignore")

# Set visual aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'


def _resolve_path(fpath: str) -> str:
    """Resolve relative file path from root or notebooks subfolder."""
    if os.path.exists(fpath):
        return fpath
    alt = os.path.join("..", fpath)
    if os.path.exists(alt):
        return alt
    return fpath


# ── 1. Data Loaders & YAML Parsers ───────────────────────────────────────────

def load_norm_failure_reports(reports_dir: str = "saved/reports") -> Dict[str, dict]:
    """Load all normalization failure YAML reports from saved/reports/."""
    rdir = _resolve_path(reports_dir)
    mapping = {
        "Bangla Deduped": os.path.join(rdir, "bangla_deduped_norm_failures.yaml"),
        "Sangraha Deduped": os.path.join(rdir, "sangraha_deduped_norm_failures.yaml"),
        "NMT Deduped": os.path.join(rdir, "nmt_deduped_nmt_norm_failures.yaml"),
        "OPUS Subtitles": os.path.join(rdir, "opus_opensubtitles_nmt_norm_failures.yaml"),
    }
    loaded = {}
    for name, path in mapping.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded[name] = yaml.safe_load(f)
    return loaded


def load_token_count_reports(reports_dir: str = "saved/reports") -> Tuple[dict, dict]:
    """Load train and eval token count YAML reports."""
    rdir = _resolve_path(reports_dir)
    train_path = os.path.join(rdir, "token_count_data_train.yaml")
    eval_path = os.path.join(rdir, "token_count_data_eval.yaml")

    train_data, eval_data = {}, {}
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = yaml.safe_load(f)
    else:
        # Fallback to any matching file
        t_files = sorted(glob.glob(os.path.join(rdir, "token_count_data_train*.yaml")))
        if t_files:
            with open(t_files[-1], "r", encoding="utf-8") as f:
                train_data = yaml.safe_load(f)

    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = yaml.safe_load(f)
    else:
        e_files = sorted(glob.glob(os.path.join(rdir, "token_count_data_eval*.yaml"))) + \
                  sorted(glob.glob(os.path.join(rdir, "token_count_bng*.yaml")))
        if e_files:
            with open(e_files[-1], "r", encoding="utf-8") as f:
                eval_data = yaml.safe_load(f)

    return train_data, eval_data


def extract_domain(fname: str) -> str:
    """Extract domain tag (bn, en, sn, nmt) from file name."""
    if fname.startswith("opus_nmt") or "_nmt_" in fname:
        return "nmt"
    elif fname.startswith("sn_") or "_sn_" in fname:
        return "sn"
    elif fname.startswith("en_") or "_en_" in fname:
        return "en"
    elif fname.startswith("bn_") or "_bn_" in fname:
        return "bn"
    return "other"


def parse_shard_metadata(report_dict: dict, split_name: str) -> pd.DataFrame:
    """Parse dataset shard metadata into a pandas DataFrame with domain mapping."""
    domain_map = {
        "bn": "Bangla Mono (bn)",
        "en": "English Mono (en)",
        "sn": "Sangraha Bangla (sn)",
        "nmt": "Translation Pairs (nmt)",
        "other": "Other Shards"
    }
    rows = []
    for fpath, stats in report_dict.get("files", {}).items():
        fname = os.path.basename(fpath)
        domain = extract_domain(fname)

        rows.append({
            "split": split_name,
            "path": fpath,
            "file": fname,
            "domain": domain,
            "domain_label": domain_map.get(domain, domain),
            "total_tokens": stats.get("total_tokens", 0),
            "total_seqs": stats.get("total_seqs", 0),
            "seq_len": stats.get("seq_len", 2048)
        })
    return pd.DataFrame(rows)


# ── 2. Structural & Statistical Audits ───────────────────────────────────────

def audit_structural_integrity(data_dirs: List[str], vocab_size: int = 48000, target_seq_len: int = 2048) -> pd.DataFrame:
    """Audit sequence shape, out-of-bounds token IDs, and dtypes across memory-mapped shards."""
    audit_records = []
    for raw_d in data_dirs:
        d = _resolve_path(raw_d)
        npy_files = sorted(glob.glob(os.path.join(d, "**", "*.npy"), recursive=True))
        for fpath in npy_files:
            fname = os.path.basename(fpath)
            try:
                arr = np.load(fpath, mmap_mode="r")
                shape = arr.shape
                dtype = str(arr.dtype)
                seq_len = shape[1] if len(shape) > 1 else 1
                total_seqs = shape[0]
                
                # Fast sample out-of-bounds check
                sample_head = arr[:100]
                sample_tail = arr[-100:]
                min_val = min(int(sample_head.min()), int(sample_tail.min()))
                max_val = max(int(sample_head.max()), int(sample_tail.max()))
                
                oob_flag = (min_val < 0) or (max_val >= vocab_size)
                shape_match = (seq_len == target_seq_len)
                
                audit_records.append({
                    "file": fname,
                    "dir": os.path.basename(d),
                    "num_seqs": total_seqs,
                    "seq_len": seq_len,
                    "dtype": dtype,
                    "min_id": min_val,
                    "max_id": max_val,
                    "oob_errors": 1 if oob_flag else 0,
                    "valid_shape": shape_match
                })
            except Exception:
                audit_records.append({
                    "file": fname,
                    "dir": os.path.basename(d),
                    "num_seqs": 0,
                    "seq_len": 0,
                    "dtype": "corrupted",
                    "min_id": -1,
                    "max_id": -1,
                    "oob_errors": 1,
                    "valid_shape": False
                })
    return pd.DataFrame(audit_records)


def audit_packing_and_composition(shard_paths: List[str], pad_id: int = 0, eos_id: int = 3, sample_seqs: int = 5000):
    """Analyze document density (EOS counts), padding ratio, and document segment lengths."""
    doc_densities = []
    padding_ratios = []
    segment_lengths = []

    for raw_fpath in shard_paths:
        fpath = _resolve_path(raw_fpath)
        if not os.path.exists(fpath):
            continue
        arr = np.load(fpath, mmap_mode="r")
        n_rows = min(len(arr), sample_seqs)
        sub = arr[:n_rows]

        # EOS count per row
        eos_counts = (sub == eos_id).sum(axis=1)
        doc_densities.extend(eos_counts.tolist())

        # Padding ratio
        pad_counts = (sub == pad_id).sum()
        padding_ratios.append(pad_counts / sub.size)

        # Segment lengths between EOS markers for first 100 rows
        for row in sub[:100]:
            eos_pos = np.where(row == eos_id)[0]
            if len(eos_pos) > 0:
                prev = 0
                for p in eos_pos:
                    seg_len = p - prev
                    if seg_len > 0:
                        segment_lengths.append(seg_len)
                    prev = p + 1

    if not doc_densities:
        doc_densities = [0]
    if not padding_ratios:
        padding_ratios = [0.0]
    if not segment_lengths:
        segment_lengths = [2048]

    return doc_densities, padding_ratios, segment_lengths


def audit_token_frequency_and_entropy(shard_paths: List[str], vocab_size: int = 48000, sample_rows: int = 10000):
    """Compute per-shard Shannon entropy, vocabulary coverage, and top-50 token dominance."""
    shard_entropies = []
    global_bincount = np.zeros(vocab_size, dtype=np.int64)
    top_50_per_shard = []

    for raw_fpath in shard_paths:
        fpath = _resolve_path(raw_fpath)
        if not os.path.exists(fpath):
            continue
        arr = np.load(fpath, mmap_mode="r")
        n = min(len(arr), sample_rows)
        flat = arr[:n].flatten()

        counts = np.bincount(flat, minlength=vocab_size)
        global_bincount += counts

        # Shannon Entropy H = - sum p * log2(p)
        probs = counts / counts.sum()
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
        shard_entropies.append(entropy)

        # Top 50 token IDs for this shard
        top50 = np.argsort(counts)[-50:][::-1]
        top_50_per_shard.append(top50)

    active_vocab = (global_bincount > 0).sum()
    vocab_coverage_pct = (active_vocab / vocab_size) * 100

    sorted_counts = np.sort(global_bincount)[::-1]
    zipf_ranks = np.arange(1, len(sorted_counts) + 1)

    return shard_entropies, vocab_coverage_pct, zipf_ranks, sorted_counts, global_bincount, top_50_per_shard


# ── 3. Clean Figure Plotting Functions (No Warnings, No Output Printing) ──────

def plot_structural_integrity(df_audit: pd.DataFrame):
    """Figure 1: Structural Integrity & Out-of-Bounds Index Scan."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(range(len(df_audit)), df_audit["seq_len"], color="#2b5c8f", linewidth=2, marker='o', markersize=3)
    axes[0].axhline(2048, color='red', linestyle='--', alpha=0.7, label='Target (2048)')
    axes[0].set_title("Sequence Length Uniformity Across All Shards")
    axes[0].set_xlabel("Shard Index")
    axes[0].set_ylabel("Sequence Length (Tokens)")
    axes[0].set_ylim(0, 2500)
    axes[0].legend(loc="lower right")

    axes[1].scatter(range(len(df_audit)), df_audit["max_id"], color="#d95f02", s=25, alpha=0.8, label="Max Token ID per Shard")
    axes[1].axhline(48000, color='black', linestyle='--', linewidth=1.5, label="Vocab Limit (48,000)")
    axes[1].set_title("Vocabulary Bound Audit (ID < 48,000 Check)")
    axes[1].set_xlabel("Shard Index")
    axes[1].set_ylabel("Max Token ID")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def plot_corpus_distribution(df_train: pd.DataFrame, df_eval: pd.DataFrame):
    """Figures 2 & 3: Token Volume & Percentage Composition (Train vs Eval)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    train_summary = df_train.groupby("domain_label")["total_tokens"].sum().reset_index()
    eval_summary = df_eval.groupby("domain_label")["total_tokens"].sum().reset_index()

    sns.barplot(data=train_summary, x="domain_label", y="total_tokens", hue="domain_label", legend=False, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Pretraining Train Token Volume by Domain (9.62B Total)")
    axes[0].set_ylabel("Total Tokens (Billions)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=15)

    for p in axes[0].patches:
        h = p.get_height()
        pct = (h / df_train["total_tokens"].sum()) * 100
        axes[0].annotate(f"{h/1e9:.2f}B\n({pct:.1f}%)",
                         (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    sns.barplot(data=eval_summary, x="domain_label", y="total_tokens", hue="domain_label", legend=False, ax=axes[1], palette="Greens_d")
    axes[1].set_title("Evaluation Token Volume by Domain (644M Total)")
    axes[1].set_ylabel("Total Tokens (Millions)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=15)

    for p in axes[1].patches:
        h = p.get_height()
        pct = (h / df_eval["total_tokens"].sum()) * 100
        axes[1].annotate(f"{h/1e6:.1f}M\n({pct:.1f}%)",
                         (p.get_x() + p.get_width() / 2., h / 2),
                         ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
    axes2[0].pie(train_summary["total_tokens"], labels=train_summary["domain_label"], autopct='%1.1f%%',
                 startangle=140, colors=sns.color_palette("Blues_r", len(train_summary)), explode=[0.03]*len(train_summary))
    axes2[0].set_title("Train Dataset Domain Share (%)")

    axes2[1].pie(eval_summary["total_tokens"], labels=eval_summary["domain_label"], autopct='%1.1f%%',
                 startangle=140, colors=sns.color_palette("Greens_r", len(eval_summary)), explode=[0.03]*len(eval_summary))
    axes2[1].set_title("Eval Dataset Domain Share (%)")

    plt.tight_layout()
    plt.show()


def plot_packing_efficiency(doc_densities: List[int], padding_ratios: List[float], segment_lengths: List[int]):
    """Figures 4, 5 & 6: Packing Efficiency, Document Density, Padding Ratio & Segment Lengths."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    max_dens = max(10, max(doc_densities)+1) if doc_densities else 10
    sns.histplot(doc_densities, bins=range(0, max_dens), ax=axes[0], color="#2b5c8f", discrete=True)
    axes[0].set_title("Document Density per Sequence (EOS Markers)")
    axes[0].set_xlabel("Number of <eos> Tokens per Sequence (2048 tokens)")
    axes[0].set_ylabel("Sequence Count")

    axes[1].bar(range(len(padding_ratios)), [p * 100 for p in padding_ratios], color="#7570b3")
    axes[1].set_title("Padding Token Ratio Across Sampled Shards")
    axes[1].set_xlabel("Shard Index")
    axes[1].set_ylabel("Padding Ratio (%)")

    sns.kdeplot(segment_lengths, ax=axes[2], color="#e7298a", fill=True, alpha=0.4)
    axes[2].set_title("Packed Document Length Distribution")
    axes[2].set_xlabel("Tokens Between <eos> Markers")
    axes[2].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_token_frequency_and_zipf(zipf_ranks: np.ndarray, sorted_counts: np.ndarray, global_bincount: np.ndarray):
    """Figures 7 & 8: Zipf's Law Log-Log Plot & Special Token Breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    nonzero_idx = sorted_counts > 0
    ranks = zipf_ranks[nonzero_idx]
    freqs = sorted_counts[nonzero_idx]

    axes[0].loglog(ranks, freqs, color="#1b9e77", linewidth=2.5, label="Token ID Frequencies")
    axes[0].set_title("Token Frequency Zipf's Law Audit (Rank vs Frequency Log-Log)")
    axes[0].set_xlabel("Token Rank (Log Scale)")
    axes[0].set_ylabel("Token Frequency (Log Scale)")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    spec_labels = ["<pad> (0)", "<unk> (1)", "<s> (2)", "</s> (3)"]
    spec_counts = [global_bincount[0], global_bincount[1], global_bincount[2], global_bincount[3]]

    sns.barplot(x=spec_labels, y=spec_counts, hue=spec_labels, legend=False, ax=axes[1], palette="Purples_d")
    axes[1].set_title("Special Token Composition Across Corpus Sample")
    axes[1].set_ylabel("Token Count")

    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_shuffling_and_uniformity(shard_entropies: List[float], df_train: pd.DataFrame):
    """Figures 9 & 10: Per-Shard Shannon Entropy & Rolling Domain Share Area Chart."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(range(1, len(shard_entropies) + 1), shard_entropies, marker='o', color='#e6ab02', linewidth=2)
    axes[0].set_title("Per-Shard Information Entropy (Shannon H)")
    axes[0].set_xlabel("Sampled Shard Index")
    axes[0].set_ylabel("Entropy (bits per token)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    domain_pivot = df_train.pivot(index="file", columns="domain_label", values="total_tokens").fillna(0)
    domain_pct = domain_pivot.div(domain_pivot.sum(axis=1), axis=0) * 100

    domain_pct.plot(kind="area", stacked=True, ax=axes[1], cmap="tab10", alpha=0.85)
    axes[1].set_title("Domain Proportion Balance Across Shards 01-48")
    axes[1].set_xlabel("Shard Index")
    axes[1].set_ylabel("Domain Token Share (%)")
    axes[1].legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_norm_failure_analysis(norm_reports: Dict[str, dict]):
    """Figures 11, 12 & 13: Unicode Normalization Failure Audit Across All 4 Corpora."""
    fail_stats = []
    for name, rep in norm_reports.items():
        fail_stats.append({
            "Corpus": name,
            "Total Failures": rep["summary"]["total_failures"],
            "Unique Tokens": rep["summary"]["unique_tokens"]
        })
    df_fail_stats = pd.DataFrame(fail_stats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    sns.barplot(data=df_fail_stats, x="Corpus", y="Total Failures", hue="Corpus", legend=False, ax=axes[0], palette="Reds_d")
    axes[0].set_title("Total Normalization Failures by Corpus Source")
    axes[0].set_ylabel("Total Failure Instances")
    axes[0].set_xlabel("")

    for p in axes[0].patches:
        h = p.get_height()
        axes[0].annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    sns.barplot(data=df_fail_stats, x="Corpus", y="Unique Tokens", hue="Corpus", legend=False, ax=axes[1], palette="Oranges_d")
    axes[1].set_title("Unique Failed Token Types by Corpus Source")
    axes[1].set_ylabel("Unique Failed Tokens")
    axes[1].set_xlabel("")

    for p in axes[1].patches:
        h = p.get_height()
        axes[1].annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

    cat_records = []
    for name, rep in norm_reports.items():
        for cat, count in rep.get("classification", {}).items():
            cat_records.append({
                "Corpus": name,
                "Category": cat,
                "Count": count
            })
    df_cats = pd.DataFrame(cat_records)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

    sns.barplot(data=df_cats, x="Category", y="Count", hue="Corpus", ax=axes2[0], palette="magma")
    axes2[0].set_title("Normalization Failure Categorization Across Corpora")
    axes2[0].set_xlabel("Failure Category")
    axes2[0].set_ylabel("Failure Count (Log Scale)")
    axes2[0].set_yscale("log")
    axes2[0].tick_params(axis='x', rotation=35)
    axes2[0].legend(title="Corpus Source")

    pivot_cats = df_cats.pivot(index="Category", columns="Corpus", values="Count").fillna(0)
    sns.heatmap(pivot_cats, annot=True, fmt=",.0f", cmap="YlOrRd", ax=axes2[1], cbar=True, linewidths=0.5)
    axes2[1].set_title("Heatmap of Failure Categories by Corpus")
    axes2[1].set_ylabel("Failure Category")
    axes2[1].set_xlabel("Corpus Source")

    plt.tight_layout()
    plt.show()

    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 9))
    axes3 = axes3.flatten()

    for i, (name, rep) in enumerate(norm_reports.items()):
        top_toks = rep.get("top_tokens", [])[:10]
        df_tok = pd.DataFrame(top_toks)
        df_tok["token_label"] = df_tok.apply(lambda r: f"{r['token']} ({r['unicode']})", axis=1)

        sns.barplot(data=df_tok, x="count", y="token_label", hue="token_label", legend=False, ax=axes3[i], palette="rocket")
        axes3[i].set_title(f"Top 10 Failures: {name}")
        axes3[i].set_xlabel("Count")
        axes3[i].set_ylabel("Token (Codepoint)")

        for p in axes3[i].patches:
            w = p.get_width()
            axes3[i].annotate(f"{int(w):,}", (w, p.get_y() + p.get_height() / 2.),
                              ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()
