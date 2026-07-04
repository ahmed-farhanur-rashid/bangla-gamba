"""
BanglaGamba — Blazing-Fast JSONL Word Counter
===============================================
Counts words, characters, and optionally tokens across any JSONL corpus.
Skips metadata — only counts the "text" field.

Usage:
  python scripts/util/count_words.py saved/data/raw/*.jsonl
  python scripts/util/count_words.py saved/data/raw/ --tokenizer saved/tokenizer/hf/
  python scripts/util/count_words.py saved/data/raw/ --workers 8 --output report.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import yaml

# ── Histogram buckets (upper bounds, exclusive) ──────────────────────────────
BUCKETS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
BUCKET_LABELS = [
    "20-50", "50-100", "100-200", "200-500", "500-1k",
    "1k-2k", "2k-5k", "5k-10k", "10k+",
]

# Reservoir size for streaming median/std estimation
RESERVOIR_SIZE = 50_000

# Unicode ranges
_BANGLA_START = 0x0980
_BANGLA_END = 0x09FF
_LATIN_START = 0x0041
_LATIN_END = 0x007A


def _bucket_idx(n: int) -> int:
    for i, b in enumerate(BUCKETS):
        if n < b:
            return i
    return len(BUCKETS)  # overflow bucket


def _is_bangla(cp: int) -> bool:
    return _BANGLA_START <= cp <= _BANGLA_END


def _is_latin(cp: int) -> bool:
    return _LATIN_START <= cp <= 0x007A or 0x0061 <= cp <= 0x007A


def count_file(args: tuple) -> dict:
    """Worker: count words/chars in one JSONL file. Returns partial stats."""
    path, seed = args
    rng = random.Random(seed)

    docs = 0
    total_words = 0
    total_chars = 0
    word_sq_sum = 0.0  # for std dev
    min_words = float("inf")
    max_words = 0

    bangla_chars = 0
    latin_chars = 0
    digit_chars = 0
    other_chars = 0

    histogram = [0] * len(BUCKET_LABELS)
    by_source = defaultdict(lambda: {"docs": 0, "words": 0, "chars": 0,
                                     "bangla": 0, "latin": 0})
    by_lang_region = defaultdict(lambda: {"docs": 0, "words": 0, "chars": 0,
                                          "bangla": 0, "latin": 0})

    # Reservoir sampling for median/std
    reservoir = []
    samples_text = []  # for tokenization (sample of raw text)

    with open(path, "rb") as f:
        for line in f:
            try:
                doc = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            text = doc.get("text", "")
            if not text:
                continue

            wc = len(text.split())
            cc = len(text)
            docs += 1
            total_words += wc
            total_chars += cc
            word_sq_sum += wc * wc
            if wc < min_words:
                min_words = wc
            if wc > max_words:
                max_words = wc

            # Histogram
            histogram[_bucket_idx(wc)] += 1

            # Character counting
            for ch in text:
                cp = ord(ch)
                if _is_bangla(cp):
                    bangla_chars += 1
                elif cp < 128:
                    if ch.isdigit():
                        digit_chars += 1
                    elif ch.isalpha():
                        latin_chars += 1
                    else:
                        other_chars += 1
                else:
                    other_chars += 1

            # Per-source
            src = doc.get("source", "unknown")
            by_source[src]["docs"] += 1
            by_source[src]["words"] += wc
            by_source[src]["chars"] += cc

            # Per-language_region
            lr = doc.get("language_region", "unknown")
            by_lang_region[lr]["docs"] += 1
            by_lang_region[lr]["words"] += wc
            by_lang_region[lr]["chars"] += cc

            # Reservoir sampling
            if len(reservoir) < RESERVOIR_SIZE:
                reservoir.append(wc)
            else:
                j = rng.randint(0, docs - 1)
                if j < RESERVOIR_SIZE:
                    reservoir[j] = wc

            # Token sample (store raw text, limit to 500)
            if len(samples_text) < 500:
                samples_text.append(text)

    # Compute bangla/latin ratios for this file
    total_cs = bangla_chars + latin_chars + digit_chars + other_chars
    bangla_ratio = bangla_chars / total_cs if total_cs > 0 else 0.0
    latin_ratio = latin_chars / total_cs if total_cs > 0 else 0.0

    # Update per-source with char ratios
    for src in by_source:
        s = by_source[src]
        s_total = s["bangla"] + s["latin"] + s["chars"] - s["bangla"] - s["latin"]
        # We didn't track per-source chars separately, approximate from totals
        s["bangla_ratio"] = 0.0  # filled in merge if needed
        s["latin_ratio"] = 0.0

    return {
        "path": path,
        "size_bytes": Path(path).stat().st_size,
        "docs": docs,
        "total_words": total_words,
        "total_chars": total_chars,
        "word_sq_sum": word_sq_sum,
        "min_words": min_words if docs > 0 else 0,
        "max_words": max_words,
        "bangla_chars": bangla_chars,
        "latin_chars": latin_chars,
        "digit_chars": digit_chars,
        "other_chars": other_chars,
        "histogram": histogram,
        "by_source": dict(by_source),
        "by_lang_region": dict(by_lang_region),
        "reservoir": reservoir,
        "samples_text": samples_text,
    }


def merge_results(partials: list[dict]) -> dict:
    """Merge all worker results into a single summary."""
    merged = {
        "total_docs": 0,
        "total_words": 0,
        "total_chars": 0,
        "word_sq_sum": 0.0,
        "min_words": float("inf"),
        "max_words": 0,
        "bangla_chars": 0,
        "latin_chars": 0,
        "digit_chars": 0,
        "other_chars": 0,
        "histogram": [0] * len(BUCKET_LABELS),
        "by_source": defaultdict(lambda: {"docs": 0, "words": 0, "chars": 0}),
        "by_lang_region": defaultdict(lambda: {"docs": 0, "words": 0, "chars": 0}),
        "reservoir": [],
        "samples_text": [],
        "files": [],
    }

    for p in partials:
        merged["total_docs"] += p["docs"]
        merged["total_words"] += p["total_words"]
        merged["total_chars"] += p["total_chars"]
        merged["word_sq_sum"] += p["word_sq_sum"]
        if p["docs"] > 0:
            if p["min_words"] < merged["min_words"]:
                merged["min_words"] = p["min_words"]
            if p["max_words"] > merged["max_words"]:
                merged["max_words"] = p["max_words"]
        merged["bangla_chars"] += p["bangla_chars"]
        merged["latin_chars"] += p["latin_chars"]
        merged["digit_chars"] += p["digit_chars"]
        merged["other_chars"] += p["other_chars"]

        for i, h in enumerate(p["histogram"]):
            merged["histogram"][i] += h

        for src, data in p["by_source"].items():
            for k in ("docs", "words", "chars"):
                merged["by_source"][src][k] += data[k]

        for lr, data in p["by_lang_region"].items():
            for k in ("docs", "words", "chars"):
                merged["by_lang_region"][lr][k] += data[k]

        # Merge reservoirs (just concatenate, final sample is large enough)
        merged["reservoir"].extend(p["reservoir"])
        merged["samples_text"].extend(p["samples_text"])

        merged["files"].append({
            "path": p["path"],
            "size_gb": round(p["size_bytes"] / (1024 ** 3), 2),
            "docs": p["docs"],
            "words": p["total_words"],
            "chars": p["total_chars"],
        })

    # Trim reservoir to target size
    rng = random.Random(42)
    if len(merged["reservoir"]) > RESERVOIR_SIZE:
        merged["reservoir"] = rng.sample(merged["reservoir"], RESERVOIR_SIZE)

    # Trim samples to 10K
    if len(merged["samples_text"]) > 10_000:
        merged["samples_text"] = rng.sample(merged["samples_text"], 10_000)

    return merged


def compute_stats(merged: dict) -> dict:
    """Compute derived stats: median, std, ratios, token estimates."""
    n = merged["total_docs"]
    if n == 0:
        return merged

    # Median and percentile from reservoir
    reservoir = sorted(merged["reservoir"])
    if reservoir:
        mid = len(reservoir) // 2
        merged["median_words_per_doc"] = reservoir[mid]
        merged["p10_words_per_doc"] = reservoir[int(len(reservoir) * 0.10)]
        merged["p25_words_per_doc"] = reservoir[int(len(reservoir) * 0.25)]
        merged["p75_words_per_doc"] = reservoir[int(len(reservoir) * 0.75)]
        merged["p90_words_per_doc"] = reservoir[int(len(reservoir) * 0.90)]
    else:
        merged["median_words_per_doc"] = 0
        merged["p10_words_per_doc"] = 0
        merged["p25_words_per_doc"] = 0
        merged["p75_words_per_doc"] = 0
        merged["p90_words_per_doc"] = 0

    # Standard deviation
    mean = merged["total_words"] / n
    variance = (merged["word_sq_sum"] / n) - (mean * mean)
    merged["std_words_per_doc"] = int(math.sqrt(max(variance, 0)))
    merged["avg_words_per_doc"] = round(mean, 1)

    # Character ratios
    total_chars = (merged["bangla_chars"] + merged["latin_chars"] +
                   merged["digit_chars"] + merged["other_chars"])
    if total_chars > 0:
        merged["char_stats"] = {
            "bangla_ratio": round(merged["bangla_chars"] / total_chars, 4),
            "latin_ratio": round(merged["latin_chars"] / total_chars, 4),
            "digit_ratio": round(merged["digit_chars"] / total_chars, 4),
            "other_ratio": round(merged["other_chars"] / total_chars, 4),
        }
    else:
        merged["char_stats"] = {}

    # Per-source stats
    by_source_out = {}
    for src, data in merged["by_source"].items():
        if data["docs"] == 0:
            continue
        total_cs = data["chars"]
        by_source_out[src] = {
            "docs": data["docs"],
            "words": data["words"],
            "chars": data["chars"],
            "avg_words": round(data["words"] / data["docs"], 1),
        }
    merged["by_source"] = by_source_out

    # Per-language_region stats
    by_lr_out = {}
    for lr, data in merged["by_lang_region"].items():
        if data["docs"] == 0:
            continue
        by_lr_out[lr] = {
            "docs": data["docs"],
            "words": data["words"],
            "chars": data["chars"],
            "avg_words": round(data["words"] / data["docs"], 1),
        }
    merged["by_lang_region"] = by_lr_out

    # Word histogram
    hist_out = {}
    for i, label in enumerate(BUCKET_LABELS):
        hist_out[label] = merged["histogram"][i]
    merged["word_histogram"] = hist_out

    # Token estimate (words × 1.49 is rough Bangla ratio)
    merged["estimated_tokens"] = int(merged["total_words"] * 1.49)

    return merged


def tokenize_sample(samples: list[str], tokenizer_path: str) -> dict:
    """Load HF tokenizer and tokenize a sample for accurate token estimates."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    except Exception as e:
        return {"error": str(e)}

    token_counts = []
    for text in samples:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))

    if not token_counts:
        return {}

    token_counts.sort()
    n = len(token_counts)
    avg = sum(token_counts) / n
    median = token_counts[n // 2]
    ratio = avg / (sum(len(t.split()) for t in samples) / n) if samples else 0

    return {
        "sample_size": n,
        "tokenizer_path": tokenizer_path,
        "avg_tokens_per_doc": round(avg, 1),
        "median_tokens_per_doc": median,
        "min_tokens_per_doc": token_counts[0],
        "max_tokens_per_doc": token_counts[-1],
        "avg_tokens_per_word": round(ratio, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Blazing-fast JSONL word counter with YAML report output."
    )
    parser.add_argument("paths", nargs="+",
                        help="JSONL files or directories to scan.")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to HF tokenizer for accurate token counting.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output YAML file path (default: stdout + auto-name).")
    args = parser.parse_args()

    # Collect all JSONL files
    jsonl_files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            jsonl_files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file() and path.suffix == ".jsonl":
            jsonl_files.append(path)
        else:
            print(f"[count] Skipping: {path}", file=sys.stderr)

    if not jsonl_files:
        print("[count] No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    print(f"[count] Found {len(jsonl_files)} JSONL file(s)", file=sys.stderr)
    for f in jsonl_files:
        size_gb = f.stat().st_size / (1024 ** 3)
        print(f"        {f.name} ({size_gb:.1f} GB)", file=sys.stderr)

    # Run workers in parallel — one per file
    tasks = [(str(f), i) for i, f in enumerate(jsonl_files)]
    partials = []
    with Pool(min(len(jsonl_files), 4)) as pool:
        for result in pool.imap_unordered(count_file, tasks):
            name = Path(result["path"]).name
            size_gb = result["size_bytes"] / (1024 ** 3)
            print(f"  ✓ {name}  ({result['docs']:,} docs, {size_gb:.1f} GB)",
                  file=sys.stderr)
            partials.append(result)

    print("[count] Merging results...", file=sys.stderr)
    merged = merge_results(partials)
    stats = compute_stats(merged)

    # Token counting
    if args.tokenizer:
        print(f"[count] Tokenizing sample with {args.tokenizer}...", file=sys.stderr)
        token_stats = tokenize_sample(stats["samples_text"], args.tokenizer)
        if "error" in token_stats:
            print(f"[count] Tokenizer error: {token_stats['error']}", file=sys.stderr)
        else:
            stats["token_sample"] = token_stats
            # Extrapolate
            ratio = token_stats.get("avg_tokens_per_word", 1.49)
            stats["estimated_tokens_by_tokenizer"] = int(stats["total_words"] * ratio)
            print(f"[count] Token ratio: {ratio:.3f} tokens/word", file=sys.stderr)

    # Clean internal fields before output
    output = {}
    key_order = [
        "total_docs", "total_words", "total_chars", "estimated_tokens",
        "estimated_tokens_by_tokenizer",
        "avg_words_per_doc", "median_words_per_doc", "std_words_per_doc",
        "min_words_per_doc", "max_words_per_doc",
        "p10_words_per_doc", "p25_words_per_doc",
        "p75_words_per_doc", "p90_words_per_doc",
        "char_stats", "word_histogram", "by_source", "by_lang_region",
        "token_sample", "files",
    ]
    for k in key_order:
        if k in stats and stats[k]:
            output[k] = stats[k]

    # Render YAML
    yaml_str = yaml.dump(output, default_flow_style=False, allow_unicode=True,
                         sort_keys=False, width=120)

    # Print to stdout
    print(yaml_str)

    # Save to file
    reports_dir = Path("saved/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        # Auto-name: count_report_<first_file_stem>.yaml
        first_name = jsonl_files[0].stem
        if len(jsonl_files) > 1:
            first_name = "multi"
        out_path = reports_dir / f"count_report_{first_name}.yaml"

    out_path.write_text(yaml_str)
    print(f"\n[count] Report saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
