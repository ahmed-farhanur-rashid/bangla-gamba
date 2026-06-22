"""
Download CulturaX Bengali split from HuggingFace.

CulturaX is the biggest clean Bangla corpus available (~1-2B tokens).
Already cleaned, deduped, CC-derived. Just download and save as JSONL.

Note: You must accept the CulturaX terms on HuggingFace first:
  https://huggingface.co/datasets/uonlp/CulturaX

Usage:
    # Check how much data there is (fast, no download)
    python scripts/download_culturax.py --check

    # Download full Bengali split
    python scripts/download_culturax.py --out saved/data/raw/culturax/

    # Download first N docs only (for testing)
    python scripts/download_culturax.py --out saved/data/raw/culturax/ --max-docs 10000
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="saved/data/raw/culturax/")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--check", action="store_true",
                         help="Just check dataset info, don't download")
    args = parser.parse_args()

    if args.check:
        print("Loading dataset info...", flush=True)
        ds = load_dataset("uonlp/CulturaX", "bn", split="train", streaming=True)
        info = ds
        # Count first 100 docs to estimate
        count = 0
        total_chars = 0
        for doc in ds:
            count += 1
            total_chars += len(doc.get("text", ""))
            if count >= 100:
                break
        avg_chars = total_chars / count
        avg_words = avg_chars / 5  # rough estimate
        print(f"Sample: {count} docs, avg {avg_chars:.0f} chars, ~{avg_words:.0f} words/doc")
        print(f"Estimated total: ~{avg_words * 1e6:.0f}M words (rough)")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "culturax_bn.jsonl"

    print("Loading CulturaX Bengali split (streaming)...", flush=True)
    ds = load_dataset("uonlp/CulturaX", "bn", split="train", streaming=True)

    count = 0
    total_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in ds:
            text = doc.get("text", "")
            if not text or len(text) < 100:
                continue

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            total_chars += len(text)

            if count % 10000 == 0:
                print(f"  {count:,} docs, {total_chars/1024/1024:.1f} MB", flush=True)

            if args.max_docs and count >= args.max_docs:
                break

    print(f"\nDone: {count:,} docs, {total_chars/1024/1024:.1f} MB", flush=True)
    print(f"Saved to: {out_path}", flush=True)
    print(f"Estimated words: ~{total_chars/5:,.0f}", flush=True)


if __name__ == "__main__":
    main()
