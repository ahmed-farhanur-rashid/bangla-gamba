"""
Count tokens in JSONL files using a HuggingFace tokenizer.

Reports total tokens, tokens per doc stats, and overall token count.

Usage:
  python util/count_tokens.py saved/data/cleaned/bangla.jsonl
  python util/count_tokens.py saved/data/cleaned/ --tokenizer saved/tokenizer/hf/
  python util/count_tokens.py saved/data/cleaned/bangla.jsonl --sample 50000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_tokenizer(path: str):
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(path)


def count_tokens(path: Path, tokenizer, sample_size: int | None = None):
    total_tokens = 0
    total_docs = 0
    token_counts = []

    with open(path) as f:
        for line in f:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "").strip()
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            n = len(tokens)
            total_tokens += n
            total_docs += 1
            token_counts.append(n)

            if total_docs % 1_000_000 == 0:
                print(f"  processed {total_docs:,} docs ({total_tokens:,} tokens)...",
                      flush=True)

    if not token_counts:
        return None

    if sample_size and sample_size < len(token_counts):
        token_counts = random.sample(token_counts, sample_size)

    token_counts.sort()
    n = len(token_counts)
    avg = total_tokens / n
    median = token_counts[n // 2]

    return {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(avg, 1),
        "median_tokens_per_doc": median,
        "min_tokens_per_doc": token_counts[0],
        "max_tokens_per_doc": token_counts[-1],
        "p10": token_counts[int(n * 0.10)],
        "p25": token_counts[int(n * 0.25)],
        "p75": token_counts[int(n * 0.75)],
        "p90": token_counts[int(n * 0.90)],
    }


def main():
    parser = argparse.ArgumentParser(description="Count tokens in JSONL files.")
    parser.add_argument("paths", nargs="+", help="JSONL files or directories.")
    parser.add_argument("--tokenizer", default="saved/tokenizer/hf/",
                        help="Path to HF tokenizer (default: saved/tokenizer/hf/).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N docs for stats (faster on huge files).")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer loaded (vocab={tokenizer.vocab_size})")

    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        elif path.is_file():
            files = [path]
        else:
            print(f"Skipping: {path}")
            continue

        for f in files:
            size_gb = f.stat().st_size / (1024 ** 3)
            print(f"\n{f.name} ({size_gb:.1f} GB)...")

            stats = count_tokens(f, tokenizer, args.sample)
            if not stats:
                print("  No docs found.")
                continue

            print(f"  Total tokens:      {stats['total_tokens']:>15,}")
            print(f"  Total docs:        {stats['total_docs']:>15,}")
            print(f"  Avg tokens/doc:    {stats['avg_tokens_per_doc']:>15}")
            print(f"  Median tokens/doc: {stats['median_tokens_per_doc']:>15,}")
            print(f"  Min / Max:         {stats['min_tokens_per_doc']:,} / {stats['max_tokens_per_doc']:,}")
            print(f"  P10 / P90:         {stats['p10']:,} / {stats['p90']:,}")
            print(f"  ~{stats['total_tokens'] / 1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
