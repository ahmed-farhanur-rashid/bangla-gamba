"""
Download BanglishRev code-mixed ecommerce reviews.
Streamed to avoid downloading image zips. Takes ALL reviews.

Usage:
  python scripts/download/01e_download_banglishrev.py
  python scripts/download/01e_download_banglishrev.py --max-docs 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import RAW_DIR, count_lines, write_doc, normalize_text, has_min_words


OUTPUT = RAW_DIR / "banglishrev.jsonl"
SOURCE = "banglishrev"
SOURCE_TYPE = "code_mixed_informal"
LANGUAGE_REGION = "BD_banglish"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs.")
    args = parser.parse_args()

    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)

    ds = load_dataset("BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset",
                       split="train", streaming=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="BanglishRev     ", unit="docs", unit_scale=True,
                   initial=existing, total=1_740_000)
        for i, row in enumerate(ds):
            if i < existing:
                continue
            if args.max_docs and i >= args.max_docs:
                break

            text = row.get("review", "")
            if not text or not isinstance(text, str):
                bar.update(1)
                continue

            text = normalize_text(text)
            if not has_min_words(text):
                bar.update(1)
                continue

            write_doc(f, text, SOURCE, SOURCE_TYPE, LANGUAGE_REGION)
            bar.update(1)
        bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 banglishrev \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
