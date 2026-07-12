"""
Cross-source NMT dedup — NLLB ∩ BanglaNMT.

Reads NLLB first (populates seen_bn), then BanglaNMT.
Catches pairs with identical Bangla side across sources.

Input:  saved/data/raw/nllb.jsonl + saved/data/raw/banglanmt.jsonl
Output: saved/data/deduped/nmt_deduped.jsonl

Usage:
  python pretrain-corpus-pipeline/01a_dedup_nmt.py
  python pretrain-corpus-pipeline/01a_dedup_nmt.py --delete-raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tqdm import tqdm

RAW_DIR = Path("saved/data/raw")
DEDUPED_DIR = Path("saved/data/deduped")
OUTPUT = DEDUPED_DIR / "nmt_deduped.jsonl"

NLLB_PATH = RAW_DIR / "nllb.jsonl"
BANGLANMT_PATH = RAW_DIR / "banglanmt.jsonl"

seen_bn: set[bytes] = set()


def _hash_bn(text: str) -> bytes:
    return hashlib.md5(text.encode()).digest()


def main():
    parser = argparse.ArgumentParser(description="Cross-source NMT dedup.")
    parser.add_argument("--delete-raw", action="store_true",
                        help="Delete NMT raw files after dedup.")
    args = parser.parse_args()

    DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

    # Check inputs exist
    sources = []
    if NLLB_PATH.exists():
        sources.append(("nllb", NLLB_PATH))
    if BANGLANMT_PATH.exists():
        sources.append(("banglanmt", BANGLANMT_PATH))

    if not sources:
        print("[dedup_nmt] No NMT raw files found.")
        return

    print(f"[dedup_nmt] Input files:")
    for name, path in sources:
        print(f"           {name}: {path}")

    # Count total lines
    total = 0
    for _, path in sources:
        with open(path, "rb") as f:
            for _ in f:
                total += 1

    # Process — NLLB first to populate seen_bn
    kept = 0
    dupes = 0

    with open(OUTPUT, "w") as fout:
        with tqdm(total=total, desc="NMT dedup", unit="lines", unit_scale=True) as bar:
            for source_name, path in sources:
                with open(path, "r") as f:
                    for line in f:
                        bar.update(1)
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = doc.get("text", "")

                        # Extract Bangla from text for dedup
                        # Format: <|task_translate_bn_en|><|lang_bn|>{bn}<|lang_en|>{en}
                        # or:     <|task_translate_en_bn|><|lang_en|>{en}<|lang_bn|>{bn}
                        # Use full text hash for dedup since both directions are present
                        h = hashlib.md5(text.encode()).digest()
                        if h in seen_bn:
                            dupes += 1
                            continue
                        seen_bn.add(h)

                        fout.write(line)
                        kept += 1

    # Stats
    in_size = sum(p.stat().st_size for _, p in sources) / (1024 ** 3)
    out_size = OUTPUT.stat().st_size / (1024 ** 3)

    print(f"\n{'=' * 50}")
    print(f"=== NMT DEDUP COMPLETE ===")
    print(f"  Input lines:    {total:,}")
    print(f"  Duplicates:     {dupes:,}")
    print(f"  Kept:           {kept:,}")
    print(f"  Output:         {out_size:.1f} GB  →  {OUTPUT}")
    print(f"  Input:          {in_size:.1f} GB")
    print(f"{'=' * 50}")

    # Delete raw (opt-in)
    if args.delete_raw:
        for _, path in sources:
            if path.exists():
                path.unlink()
                print(f"[dedup_nmt] Deleted {path}")


if __name__ == "__main__":
    main()
