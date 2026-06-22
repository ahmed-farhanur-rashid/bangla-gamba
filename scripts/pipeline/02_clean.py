"""
BanglaGamba — Clean & Deduplicate
===================================
Two-pass streaming approach:
  Pass 1: Filter all raw/*.jsonl → temp filtered file (no memory buildup)
  Pass 2: MinHash dedup the temp file → final output

Usage:
  python scripts/pipeline/02_clean.py
  python scripts/pipeline/02_clean.py --skip-dedup
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
TEMP_PATH = CLEANED_DIR / ".filtered_tmp.jsonl"


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


def count_lines(path: Path) -> int:
    """Fast line count using binary read."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def pass_filter(raw_files: list[Path], total: int) -> tuple[int, int, int]:
    """Pass 1: Stream raw files, filter, write to temp file. Returns counts."""
    kept = 0
    length_rejected = 0
    quality_rejected = 0

    with open(TEMP_PATH, "w") as fout:
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

    return kept, length_rejected, quality_rejected


def pass_dedup(total_filtered: int) -> tuple[int, int]:
    """Pass 2: Stream temp file, MinHash dedup, write final output. Returns (kept, rejected)."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("[clean] WARNING: datasketch not installed, skipping deduplication")
        # No dedup — just rename temp to output
        TEMP_PATH.rename(OUTPUT_PATH)
        return total_filtered, 0

    print("[clean] Running MinHash deduplication (num_perm=128, threshold=0.80)...")
    lsh = MinHashLSH(threshold=0.80, num_perm=128)
    dedup_rejected = 0
    doc_id = 0

    def get_minhash(text: str) -> MinHash:
        m = MinHash(num_perm=128)
        words = text.split()
        for i in range(len(words) - 4):
            gram = " ".join(words[i:i + 5])
            m.update(gram.encode("utf-8"))
        return m

    with open(TEMP_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
        with tqdm(total=total_filtered, desc="Dedup     ", unit="docs", unit_scale=True) as bar:
            for line in fin:
                bar.update(1)
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = doc["text"]
                mh = get_minhash(text)
                if lsh.query(mh):
                    dedup_rejected += 1
                    continue

                lsh.insert(str(doc_id), mh)
                doc["doc_id"] = doc_id
                doc_id += 1
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return doc_id, dedup_rejected


def run_clean(skip_dedup: bool = False, delete_raw: bool = False):
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

    # Pass 1: Filter → temp file
    print("[clean] Pass 1: Filtering documents...")
    kept, length_rejected, quality_rejected = pass_filter(raw_files, total)
    tqdm.write(f"  After filtering: {kept:,} / {total:,}")

    # Pass 2: Dedup → final output
    if skip_dedup:
        print("[clean] Skipping dedup (--skip-dedup)")
        TEMP_PATH.rename(OUTPUT_PATH)
        final_count = kept
        dedup_rejected = 0
    else:
        print("[clean] Pass 2: Deduplicating...")
        final_count, dedup_rejected = pass_dedup(kept)
        tqdm.write(f"  After dedup: {final_count:,}")

    # Clean up temp file
    if TEMP_PATH.exists():
        TEMP_PATH.unlink()

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
    print(f"Dedup rejected:       {dedup_rejected:,}")
    print(f"Kept:                 {final_count:,}  ({final_count / max(total, 1) * 100:.1f}%)")
    print(f"Output:               {output_size:.1f} GB  →  {OUTPUT_PATH}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate raw corpus.")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip MinHash deduplication (faster, for testing).")
    parser.add_argument("--delete-raw", action="store_true",
                        help="Delete raw/ directory after cleaning.")
    args = parser.parse_args()
    run_clean(skip_dedup=args.skip_dedup, delete_raw=args.delete_raw)


if __name__ == "__main__":
    main()
