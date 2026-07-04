"""
Dedup Sangraha Verified against the Bangla base (wiki + titullm).

Builds hash set from deduped/bangla_deduped.jsonl (read-only).
Streams sangraha, drops duplicates found in base.
No normalization — just hash dedup. Run 01d_bn_normalize.py after.

Input:  saved/data/deduped/bangla_deduped.jsonl  (base, read-only)
        saved/data/raw/sangraha_verified_bn.jsonl
Output: saved/data/deduped/sangraha_deduped.jsonl

Usage:
  python pretrain-corpus-pipeline/01c_dedup_sangraha.py
  python pretrain-corpus-pipeline/01c_dedup_sangraha.py --max-words 2_000_000_000
  python pretrain-corpus-pipeline/01c_dedup_sangraha.py --dry-run
  python pretrain-corpus-pipeline/01c_dedup_sangraha.py --max-docs 1_000_000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

DEDUPED_DIR = Path("saved/data/deduped")
RAW_DIR = Path("saved/data/raw")

BASE_PATH = DEDUPED_DIR / "bangla_deduped.jsonl"
SANGRAHA_PATH = RAW_DIR / "sangraha_verified_bn.jsonl"
OUTPUT = DEDUPED_DIR / "sangraha_deduped.jsonl"


def normalize_for_hash(text: str) -> bytes:
    """Normalize text before hashing so trivial formatting/encoding
    differences (whitespace, Unicode form) don't defeat exact dedup."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("utf-8")


def load_base_hashes(base_path: Path) -> set[bytes]:
    """Load all text hashes from the base deduped file."""
    hashes = set()
    if not base_path.exists():
        print(f"[dedup_sangraha] WARNING: base file not found: {base_path}")
        return hashes

    print(f"[dedup_sangraha] Counting base docs ...")
    total = 0
    with open(base_path, "rb") as f:
        for _ in f:
            total += 1

    print(f"[dedup_sangraha] Loading base hashes from {base_path} ...")
    with open(base_path, "r") as f:
        for line in tqdm(f, total=total, desc="Loading base hashes", unit="docs", unit_scale=True):
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                h = normalize_for_hash(text)
                if h:
                    hashes.add(hashlib.sha256(h).digest())
            except json.JSONDecodeError:
                continue
    print(f"[dedup_sangraha] Base hashes loaded: {len(hashes):,}")
    return hashes


def main():
    parser = argparse.ArgumentParser(description="Dedup Sangraha against Bangla base.")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Cap output to N words (stops early when reached).")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap output to N docs (stops early when reached).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report overlap stats without writing output.")
    args = parser.parse_args()

    DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_PATH.exists():
        print(f"[dedup_sangraha] Base file not found: {BASE_PATH}")
        print(f"  Run 01b_dedup_mono_bn.py first.")
        return

    if not SANGRAHA_PATH.exists():
        print(f"[dedup_sangraha] Sangraha file not found: {SANGRAHA_PATH}")
        return

    # Step 1: Load base hashes
    base_hashes = load_base_hashes(BASE_PATH)

    # Step 2: Stream sangraha, dedup against base + within-sangraha
    print(f"\n[dedup_sangraha] Processing {SANGRAHA_PATH} ...")

    # Count sangraha lines for progress bar
    print("[dedup_sangraha] Counting sangraha docs ...")
    sangraha_total = 0
    with open(SANGRAHA_PATH, "rb") as f:
        for _ in f:
            sangraha_total += 1
    print(f"[dedup_sangraha] Sangraha docs: {sangraha_total:,}")

    seen_hashes: set[bytes] = set(base_hashes)
    kept = 0
    empty_text = 0
    dupes_base = 0
    dupes_self = 0
    words_written = 0

    fout = None
    if not args.dry_run:
        fout = open(OUTPUT, "w")

    with tqdm(total=sangraha_total, desc="Sangraha dedup", unit="docs", unit_scale=True) as bar:
        with open(SANGRAHA_PATH, "r") as f:
            for line in f:
                bar.update(1)

                # Check limits
                if args.max_docs and kept >= args.max_docs:
                    break
                if args.max_words and words_written >= args.max_words:
                    break

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = doc.get("text", "")
                hash_bytes = normalize_for_hash(text)
                if not hash_bytes:
                    empty_text += 1
                    continue

                h = hashlib.sha256(hash_bytes).digest()

                if h in base_hashes:
                    dupes_base += 1
                    continue
                if h in seen_hashes:
                    dupes_self += 1
                    continue

                seen_hashes.add(h)
                kept += 1
                wc = len(text.split())
                words_written += wc

                if fout:
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    if fout:
        fout.close()

    # Stats
    sangraha_size = SANGRAHA_PATH.stat().st_size / (1024 ** 3)
    out_size = OUTPUT.stat().st_size / (1024 ** 3) if not args.dry_run else 0

    print(f"\n{'=' * 50}")
    print(f"=== SANGRAHA DEDUP COMPLETE ===")
    print(f"  Sangraha docs:  {sangraha_total:,}")
    print(f"  Empty text:     {empty_text:,}")
    print(f"  Dupes vs base:  {dupes_base:,}")
    print(f"  Dupes vs self:  {dupes_self:,}")
    print(f"  Kept:           {kept:,}")
    print(f"  Words kept:     {words_written:,}")
    if not args.dry_run:
        print(f"  Output:         {out_size:.1f} GB  →  {OUTPUT}")
    print(f"  Sangraha size:  {sangraha_size:.1f} GB")
    if args.max_words:
        print(f"  Word cap:       {args.max_words:,}  (reached: {words_written >= args.max_words})")
    if args.max_docs:
        print(f"  Doc cap:        {args.max_docs:,}  (reached: {kept >= args.max_docs})")
    if args.dry_run:
        print(f"  *** DRY RUN — no output written ***")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
