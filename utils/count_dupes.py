"""
Count exact duplicates in JSONL files by SHA-256 hash.

Usage:
  python util/count_dupes.py saved/data/raw/titullm.jsonl
  python util/count_dupes.py saved/data/cleaned/bangla.jsonl --field source
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def count_dupes(path: Path, field: str | None = None):
    hashes: dict[bytes, int] = Counter()
    total = 0

    with open(path) as f:
        for line in f:
            total += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "")
            h = hashlib.sha256(text.encode()).digest()
            hashes[h] += 1
            if total % 1_000_000 == 0:
                print(f"  processed {total:,}...", flush=True)

    unique = len(hashes)
    dupes = sum(v - 1 for v in hashes.values() if v > 1)
    dupe_pct = (dupes / total * 100) if total > 0 else 0

    print(f"\n  Total docs:    {total:>12,}")
    print(f"  Unique docs:   {unique:>12,}")
    print(f"  Duplicates:    {dupes:>12,}  ({dupe_pct:.1f}%)")

    if field:
        print(f"\n  Top duplicate groups by '{field}':")
        groups = Counter()
        for h, count in hashes.items():
            if count > 1:
                groups[count] += 1
        for count, num_groups in groups.most_common(10):
            print(f"    {count}x duplicated: {num_groups:,} groups")


def main():
    parser = argparse.ArgumentParser(description="Count exact duplicates in JSONL.")
    parser.add_argument("path", help="JSONL file to check.")
    parser.add_argument("--field", default=None,
                        help="Extra field to group by (e.g. 'source').")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    size_gb = path.stat().st_size / (1024 ** 3)
    print(f"Checking {path.name} ({size_gb:.1f} GB)...")
    count_dupes(path, args.field)


if __name__ == "__main__":
    main()
