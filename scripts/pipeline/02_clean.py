"""
BanglaGamba — Clean Corpus
============================
Single-pass streaming: filter all raw/*.jsonl → corpus_cleaned.jsonl.
Deduplication is handled separately by 02b_dedup.py.

Usage:
  python scripts/pipeline/02_clean.py
  python scripts/pipeline/02_clean.py --delete-raw
"""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from pathlib import Path

from tqdm import tqdm


RAW_DIR = Path("saved/data/raw")
CLEANED_DIR = Path("saved/data/cleaned")
OUTPUT_PATH = CLEANED_DIR / "corpus_cleaned.jsonl"


def normalize_text(text: str) -> str:
    """Apply NFC unicode normalization, optional Bangla normalizer, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    try:
        from bnunicodenormalizer import Normalizer
        norm = Normalizer()
        text = norm.normalize(text)
        if isinstance(text, dict):
            text = text.get("normalized", text.get("text", ""))
    except (ImportError, Exception):
        pass
    text = " ".join(text.split())
    return text


def should_skip_quality(text: str, language_region: str) -> bool:
    """Apply script/quality filter only to Bangla-script sources."""
    if language_region != "BD_WB_mix":
        return False

    if len(text) > 0:
        ascii_count = sum(1 for ch in text if ord(ch) < 128)
        if ascii_count / len(text) > 0.45:
            return True

    if len(text) > 0:
        punct_count = sum(1 for ch in text if ch.isascii() and not ch.isalnum() and not ch.isspace())
        if punct_count / len(text) > 0.30:
            return True

    return False


def run_clean(delete_raw: bool = False):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print("[clean] No raw JSONL files found in saved/data/raw/")
        return

    print(f"[clean] Found {len(raw_files)} raw files:")
    for f in raw_files:
        print(f"        {f.name}")

    # Count total lines
    print("[clean] Counting total documents...")
    total = 0
    for f in raw_files:
        with open(f, "rb") as fh:
            for _ in fh:
                total += 1

    # Filter → output
    print("[clean] Filtering documents...")
    kept = 0
    length_rejected = 0
    quality_rejected = 0

    with open(OUTPUT_PATH, "w") as fout:
        with tqdm(total=total, desc="Filtering", unit="docs", unit_scale=True) as bar:
            for raw_file in raw_files:
                with open(raw_file, "r") as f:
                    for line in f:
                        bar.update(1)
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = doc.get("text", "")
                        source_type = doc.get("source_type", "")
                        language_region = doc.get("language_region", "")

                        # Length filter
                        min_words = 10 if source_type == "parallel_bn_en" else 20
                        if len(text.split()) < min_words:
                            length_rejected += 1
                            continue

                        # Unicode normalization
                        text = normalize_text(text)
                        doc["text"] = text

                        # Quality filter (Bangla-script only)
                        if should_skip_quality(text, language_region):
                            quality_rejected += 1
                            continue

                        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        kept += 1

    output_size = OUTPUT_PATH.stat().st_size / (1024 ** 3)

    # Delete raw (opt-in only)
    if delete_raw:
        raw_size = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file()) / (1024 ** 3)
        shutil.rmtree(RAW_DIR)
        print(f"[clean] Deleted {RAW_DIR} — freed {raw_size:.1f} GB")

    # Summary
    print("\n" + "=" * 50)
    print("=== CLEANING COMPLETE ===")
    print(f"Total read:           {total:,}")
    print(f"Length rejected:      {length_rejected:,}")
    print(f"Quality rejected:     {quality_rejected:,}")
    print(f"Kept:                 {kept:,}  ({kept / max(total, 1) * 100:.1f}%)")
    print(f"Output:               {output_size:.1f} GB  →  {OUTPUT_PATH}")
    print(f"Next step:            python scripts/pipeline/02b_dedup.py")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Clean raw corpus (filter + normalize).")
    parser.add_argument("--delete-raw", action="store_true",
                        help="Delete raw/ directory after cleaning.")
    args = parser.parse_args()
    run_clean(delete_raw=args.delete_raw)


if __name__ == "__main__":
    main()
