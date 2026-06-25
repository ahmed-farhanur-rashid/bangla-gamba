"""
Download English from FineWeb (sample-10BT).
Streams and stops once word budget is hit.

Output (v3 schema):
  {"text": "<|lang_en|>...", "source": "fineweb_edu", "source_type": "web_english",
   "language_region": "EN", "word_count": N}

Usage:
  python scripts/downloaders/02a_download_english.py
  python scripts/downloaders/02a_download_english.py --word-budget 2_000_000_000
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from _common import (
    RAW_DIR, CLEANED_DIR, LANG_EN, count_lines, write_doc,
    nfc, wc,
)

OUTPUT = CLEANED_DIR / "english.jsonl"
SOURCE = "fineweb_edu"
SOURCE_TYPE = "web_english"
LANGUAGE_REGION = "EN"
DEFAULT_WORD_BUDGET = 1_000_000_000
AVG_WORDS_PER_DOC = 300


def download_fineweb(word_budget: int):
    from datasets import load_dataset

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    existing = count_lines(OUTPUT)

    if existing > 0:
        estimated_words = existing * AVG_WORDS_PER_DOC
        if estimated_words >= word_budget:
            print(f"  \u21b7 fineweb already complete (~{estimated_words:,} words estimated), skipping")
            return

    print("── Downloading FineWeb sample-10BT ──")
    print(f"   Word budget: {word_budget:,}")
    print(f"   Existing: {existing:,} docs")
    print(f"   Streaming — will stop when budget is hit\n")

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    total_words = existing * AVG_WORDS_PER_DOC
    docs = 0
    skipped = 0

    with open(OUTPUT, "a") as f:
        for row in tqdm(ds, desc="FineWeb", unit=" docs"):
            if total_words >= word_budget:
                print(f"\n   Budget reached.")
                break

            text = (row.get("text") or "").strip()

            if not text:
                skipped += 1
                continue

            text = nfc(text)
            word_count = wc(text)

            if word_count < 50 or word_count > 100_000:
                skipped += 1
                continue

            write_doc(f, f"{LANG_EN}{text}", SOURCE, SOURCE_TYPE,
                      LANGUAGE_REGION, word_count)

            total_words += word_count
            docs += 1

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\nDone.")
    print(f"  Documents      : {docs:>12,}")
    print(f"  Words collected: {total_words:>12,}")
    print(f"  Skipped        : {skipped:>12,}")
    print(f"  Output         : {OUTPUT}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download English from FineWeb.")
    parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET,
                        help=f"Word budget (default: {DEFAULT_WORD_BUDGET:,}).")
    args = parser.parse_args()
    download_fineweb(args.word_budget)


if __name__ == "__main__":
    main()
