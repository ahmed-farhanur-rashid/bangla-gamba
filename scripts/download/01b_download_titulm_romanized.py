"""
Download TituLM Romanized Bangla corpus.
Full dataset ~5GB, streamed. Supports mid-download resumption.

Usage:
  python scripts/download/01b_download_titulm_romanized.py
  python scripts/download/01b_download_titulm_romanized.py --max-docs 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import RAW_DIR, count_lines, write_doc, normalize_text, has_min_words


OUTPUT = RAW_DIR / "titulm_romanized.jsonl"
SOURCE = "titulm_romanized"
SOURCE_TYPE = "romanized_bangla"
LANGUAGE_REGION = "BD_banglish"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs.")
    args = parser.parse_args()

    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)

    ds = load_dataset("hishab/titulm-bangla-corpus", data_dir="romanized",
                       split="train", streaming=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="TituLM Romanized", unit="docs", unit_scale=True, initial=existing)
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
    print(f"  \u2713 titulm_romanized \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
