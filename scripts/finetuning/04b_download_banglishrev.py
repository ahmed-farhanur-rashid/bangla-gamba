"""
Download BanglishRev code-mixed ecommerce reviews.
Downloads reviews v1.json directly from HF Hub (avoids 50GB of image zips).
Takes ALL reviews.

Usage:
  python scripts/download/01e_download_banglishrev.py
  python scripts/download/01e_download_banglishrev.py --max-docs 5000
"""

from __future__ import annotations

import argparse
import json
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

    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)
    if args.max_docs and existing >= args.max_docs:
        print(f"  \u21b7 banglishrev already complete ({existing:,} docs), skipping")
        return

    # Download reviews v1.json (2GB) directly
    print("[banglishrev] Downloading reviews v1.json from HF Hub...")
    json_path = hf_hub_download(
        "BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset",
        "reviews v1.json",
        repo_type="dataset",
    )
    print(f"[banglishrev] Downloaded to cache: {json_path}")

    # Parse nested JSON: list of products, each with "Reviews" list
    print("[banglishrev] Parsing reviews...")
    with open(json_path, "r", encoding="utf-8") as jf:
        products = json.load(jf)

    # Count total reviews for progress bar
    total_reviews = sum(len(p.get("Reviews", [])) for p in products)
    print(f"[banglishrev] Found {len(products):,} products, {total_reviews:,} reviews")

    written = existing
    skipped_existing = existing
    bar = tqdm(desc="BanglishRev     ", unit="docs", unit_scale=True,
               initial=existing, total=total_reviews)

    with open(OUTPUT, "a") as f:
        for product in products:
            for review in product.get("Reviews", []):
                bar.update(1)

                if skipped_existing > 0:
                    skipped_existing -= 1
                    continue
                if args.max_docs and written >= args.max_docs:
                    break

                text = review.get("Review Content", "")
                if not text or not isinstance(text, str):
                    continue

                text = normalize_text(text)
                if not has_min_words(text):
                    continue

                write_doc(f, text, SOURCE, SOURCE_TYPE, LANGUAGE_REGION)
                written += 1

            if args.max_docs and written >= args.max_docs:
                break

    bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 banglishrev \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
