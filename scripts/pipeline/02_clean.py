"""
BanglaGamba — Clean & Deduplicate
===================================
Clean all saved/data/raw/*.jsonl → single merged file.
Delete saved/data/raw/ after.

Usage:
  python scripts/pipeline/02_clean.py
  python scripts/pipeline/02_clean.py --skip-dedup
  python scripts/pipeline/02_clean.py --no-delete-raw
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

    # ASCII ratio check
    if len(text) > 0:
        ascii_count = sum(1 for ch in text if ord(ch) < 128)
        if ascii_count / len(text) > 0.45:
            return True

    # Punctuation ratio check
    if len(text) > 0:
        punct_count = sum(1 for ch in text if ch.isascii() and not ch.isalnum() and not ch.isspace())
        if punct_count / len(text) > 0.30:
            return True

    return False


def run_clean(skip_dedup: bool = False, no_delete_raw: bool = False):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Find all raw JSONL files
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print("[clean] No raw JSONL files found in saved/data/raw/")
        return

    print(f"[clean] Found {len(raw_files)} raw files:")
    for f in raw_files:
        print(f"        {f.name}")

    # Count total lines for progress bar
    print("[clean] Counting total documents...")
    total = 0
    for f in raw_files:
        with open(f, "rb") as fh:
            for _ in fh:
                total += 1

    # Stage 1-3: Filter
    print("[clean] Filtering documents...")
    kept = 0
    length_rejected = 0
    quality_rejected = 0
    docs = []

    with tqdm(total=total, desc="Cleaning", unit="docs", unit_scale=True) as bar:
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

                    # Stage 1: Length filter
                    min_words = 10 if source_type == "parallel_bn_en" else 20
                    if len(text.split()) < min_words:
                        length_rejected += 1
                        continue

                    # Stage 2: Unicode normalization
                    text = normalize_text(text)
                    doc["text"] = text

                    # Stage 3: Quality filter (Bangla-script only)
                    if should_skip_quality(text, language_region):
                        quality_rejected += 1
                        continue

                    docs.append(doc)
                    kept += 1

    tqdm.write(f"  After filtering: {kept:,} / {total:,}")

    # Stage 4: MinHash deduplication
    dedup_rejected = 0
    if not skip_dedup:
        try:
            from datasketch import MinHash, MinHashLSH
            print("[clean] Running MinHash deduplication (num_perm=128, threshold=0.80)...")

            lsh = MinHashLSH(threshold=0.80, num_perm=128)
            unique_docs = []

            def get_minhash(text: str) -> MinHash:
                m = MinHash(num_perm=128)
                words = text.split()
                for i in range(len(words) - 4):
                    gram = " ".join(words[i:i + 5])
                    m.update(gram.encode("utf-8"))
                return m

            with tqdm(total=len(docs), desc="Dedup     ", unit="docs", unit_scale=True) as bar:
                for doc in docs:
                    text = doc["text"]
                    mh = get_minhash(text)
                    if lsh.query(mh):
                        dedup_rejected += 1
                    else:
                        lsh.insert(text[:64], mh)
                        unique_docs.append(doc)
                    bar.update(1)

            docs = unique_docs
            tqdm.write(f"  After dedup: {len(docs):,}")
        except ImportError:
            print("[clean] WARNING: datasketch not installed, skipping deduplication")

    # Stage 5: Shuffle and write
    print("[clean] Shuffling and writing...")
    import random
    random.seed(42)
    random.shuffle(docs)

    with open(OUTPUT_PATH, "w") as f:
        for i, doc in enumerate(docs):
            doc["doc_id"] = i
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    output_size = OUTPUT_PATH.stat().st_size / (1024 ** 3)

    # Stage 6: Delete raw
    if not no_delete_raw:
        raw_size = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file()) / (1024 ** 3)
        shutil.rmtree(RAW_DIR)
        print(f"[clean] Deleted {RAW_DIR} — freed {raw_size:.1f} GB")

    # Summary
    print("\n" + "=" * 50)
    print("=== CLEANING COMPLETE ===")
    print(f"Total read:           {total:,}")
    print(f"Length rejected:      {length_rejected:,}")
    print(f"Quality rejected:     {quality_rejected:,}")
    print(f"Dedup rejected:       {dedup_rejected:,}")
    print(f"Kept:                 {len(docs):,}  ({len(docs) / max(total, 1) * 100:.1f}%)")
    print(f"Output:               {output_size:.1f} GB  →  {OUTPUT_PATH}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate raw corpus.")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip MinHash deduplication (faster, for testing).")
    parser.add_argument("--no-delete-raw", action="store_true",
                        help="Keep raw/ directory after cleaning.")
    args = parser.parse_args()
    run_clean(skip_dedup=args.skip_dedup, no_delete_raw=args.no_delete_raw)


if __name__ == "__main__":
    main()
