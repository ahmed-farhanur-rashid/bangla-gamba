#!/usr/bin/env python3
"""
One-time Bangla normalization pass using bnunicodenormalizer.

Downloads now use NFC only (fast). Run this script after all downloads
complete to apply full Bangla-specific normalization to raw JSONL files.

Usage:
    python scripts/pipeline/bn_normalize.py                          # default: all files
    python scripts/pipeline/bn_normalize.py --files saved/data/raw/titullm.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

try:
    from bnunicodenormalizer import Normalizer
except ImportError:
    print("pip install bnunicodenormalizer")
    sys.exit(1)

RAW_DIR = Path("saved/data/raw")

DEFAULT_FILES = [
    "saved/data/raw/titullm.jsonl",
    "saved/data/raw/wiki_bangla.jsonl",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply bnunicodenormalizer to Bangla text in raw JSONL files."
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="JSONL files to normalize (default: titullm + wiki_bangla)",
    )
    p.add_argument(
        "--prefix",
        default="<|lang_bn|>",
        help="Token prefix to strip/re-add (default: <|lang_bn|>)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    return p.parse_args()


def normalize_words(norm: Normalizer, text: str, prefix: str) -> str:
    """Strip prefix, normalize each word, re-add prefix."""
    stripped = text
    had_prefix = stripped.startswith(prefix)
    if had_prefix:
        stripped = stripped[len(prefix):].lstrip()

    words = stripped.split()
    normalized = []
    for w in words:
        result = norm(w)
        if isinstance(result, dict):
            n = result.get("normalized") or result.get("text") or w
            normalized.append(n)
        else:
            normalized.append(result if result else w)

    out = " ".join(normalized)
    if had_prefix:
        out = prefix + " " + out
    return out


def process_file(path: Path, norm: Normalizer, prefix: str, dry_run: bool) -> dict:
    """Process one JSONL file in-place using a temp file."""
    stats = {"read": 0, "normalized": 0, "skipped": 0, "bytes": path.stat().st_size}

    if dry_run:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stats["read"] += 1
                doc = json.loads(line)
                if doc.get("language_region", "").startswith("bn"):
                    stats["normalized"] += 1
                else:
                    stats["skipped"] += 1
        return stats

    tmp = path.with_suffix(".tmp")
    with open(path, encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            stats["read"] += 1
            doc = json.loads(line)

            if not doc.get("language_region", "").startswith("bn"):
                stats["skipped"] += 1
                fout.write(line)
                continue

            doc["text"] = normalize_words(norm, doc["text"], prefix)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            stats["normalized"] += 1

    shutil.move(str(tmp), str(path))
    return stats


def main():
    args = parse_args()
    norm = Normalizer()

    total = {"read": 0, "normalized": 0, "skipped": 0, "bytes": 0}
    start = time.time()

    for fname in args.files:
        path = Path(fname)
        if not path.exists():
            print(f"  skipping {path} (not found)")
            continue

        s = process_file(path, norm, args.prefix, args.dry_run)
        mb = s["bytes"] / 1e6
        print(f"  {path.name}: {s['normalized']:,} normalized, "
              f"{s['skipped']:,} skipped ({mb:.0f} MB)")

        for k in total:
            total[k] += s[k]

    elapsed = time.time() - start
    mb = total["bytes"] / 1e6
    print(f"\nTotal: {total['normalized']:,} Bangla docs normalized, "
          f"{total['skipped']:,} non-Bangla skipped ({mb:.0f} MB)")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
