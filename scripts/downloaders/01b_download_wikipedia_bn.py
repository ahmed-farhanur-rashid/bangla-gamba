"""
Download Bengali Wikipedia.
Small ~90K articles, streamed.

Output (v3 schema):
  {"text": "<|lang_bn|>...", "source": "wiki_bangla", "source_type": "web_bangla",
   "language_region": "BD", "word_count": N}

Usage:
  python scripts/downloaders/01b_download_wikipedia_bn.py
  python scripts/downloaders/01b_download_wikipedia_bn.py --max-docs 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import (
    RAW_DIR, LANG_BN, count_lines, write_doc,
    normalize_doc, wc, has_min_words,
)


OUTPUT = RAW_DIR / "wiki_bangla.jsonl"
SOURCE = "wiki_bangla"
SOURCE_TYPE = "web_bangla"
LANGUAGE_REGION = "BD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs.")
    args = parser.parse_args()

    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)

    ds = load_dataset("wikimedia/wikipedia", "20231101.bn",
                       split="train", streaming=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="Wikipedia BN    ", unit="docs", unit_scale=True,
                   initial=existing)
        written = 0
        skip = existing
        for row in ds:
            if args.max_docs and written + existing >= args.max_docs:
                break
            text = normalize_doc(row.get("text", ""))
            if not has_min_words(text):
                continue
            if skip > 0:
                skip -= 1
                bar.update(1)
                continue
            n = wc(text)
            write_doc(f, f"{LANG_BN}{text}", SOURCE, SOURCE_TYPE,
                      LANGUAGE_REGION, n)
            written += 1
            bar.update(1)
        bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 wiki_bangla \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
