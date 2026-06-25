"""
Cross-source monolingual dedup — TituLLM ∩ Wiki Bangla.

Exact dedup via SHA-256 on text content.
Catches documents that appear in both TituLLM and Wikipedia.

Input:  saved/data/raw/titullm.jsonl + saved/data/raw/wiki_bangla.jsonl
Output: saved/data/cleaned/bangla.jsonl

Usage:
  python scripts/pipeline/02b_dedup_mono.py
  python scripts/pipeline/02b_dedup_mono.py --delete-raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tqdm import tqdm

RAW_DIR = Path("saved/data/raw")
CLEANED_DIR = Path("saved/data/cleaned")
OUTPUT = CLEANED_DIR / "bangla.jsonl"

SOURCES = [
    ("titullm", RAW_DIR / "titullm.jsonl"),
    ("wiki_bangla", RAW_DIR / "wiki_bangla.jsonl"),
]


def main():
    parser = argparse.ArgumentParser(description="Cross-source monolingual dedup.")
    parser.add_argument("--delete-raw", action="store_true",
                        help="Delete Bangla raw files after dedup.")
    args = parser.parse_args()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Check inputs exist
    existing = [(name, path) for name, path in SOURCES if path.exists()]

    if not existing:
        print("[dedup_mono] No Bangla raw files found.")
        return

    print(f"[dedup_mono] Input files:")
    for name, path in existing:
        print(f"             {name}: {path}")

    # Count total lines
    total = 0
    for _, path in existing:
        with open(path, "rb") as f:
            for _ in f:
                total += 1

    # Exact dedup via SHA-256
    seen_hashes: set[bytes] = set()
    kept = 0
    dupes = 0

    with open(OUTPUT, "w") as fout:
        with tqdm(total=total, desc="Mono dedup", unit="docs", unit_scale=True) as bar:
            for source_name, path in existing:
                with open(path, "r") as f:
                    for line in f:
                        bar.update(1)
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = doc.get("text", "")
                        h = hashlib.sha256(text.encode()).digest()

                        if h in seen_hashes:
                            dupes += 1
                            continue
                        seen_hashes.add(h)

                        fout.write(line)
                        kept += 1

    # Stats
    in_size = sum(p.stat().st_size for _, p in existing) / (1024 ** 3)
    out_size = OUTPUT.stat().st_size / (1024 ** 3)

    print(f"\n{'=' * 50}")
    print(f"=== MONO DEDUP COMPLETE ===")
    print(f"  Input docs:     {total:,}")
    print(f"  Duplicates:     {dupes:,}")
    print(f"  Kept:           {kept:,}")
    print(f"  Output:         {out_size:.1f} GB  →  {OUTPUT}")
    print(f"  Input:          {in_size:.1f} GB")
    print(f"{'=' * 50}")

    if args.delete_raw:
        for _, path in existing:
            if path.exists():
                path.unlink()
                print(f"[dedup_mono] Deleted {path}")


if __name__ == "__main__":
    main()
