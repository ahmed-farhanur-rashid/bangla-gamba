"""
Download Bengali Wikipedia.
Small ~90K articles, loaded fully (not streamed).

Usage:
  python scripts/download/01c_download_wikipedia.py
  python scripts/download/01c_download_wikipedia.py --max-docs 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import RAW_DIR, count_lines, write_doc, normalize_text, has_min_words


OUTPUT = RAW_DIR / "wikipedia_bn.jsonl"
SOURCE = "wikipedia_bn"
SOURCE_TYPE = "encyclopedic"
LANGUAGE_REGION = "BD_WB_mix"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs.")
    args = parser.parse_args()

    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)

    ds = load_dataset("wikimedia/wikipedia", "20231101.bn",
                       split="train", streaming=False)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="Wikipedia BN    ", unit="docs", unit_scale=True,
                   initial=existing, total=len(ds))
        for i, row in enumerate(ds):
            if i < existing:
                continue
            if args.max_docs and i >= args.max_docs:
                break
            text = normalize_text(row.get("text", ""))
            if has_min_words(text):
                write_doc(f, text, SOURCE, SOURCE_TYPE, LANGUAGE_REGION)
            bar.update(1)
        bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 wikipedia_bn \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
