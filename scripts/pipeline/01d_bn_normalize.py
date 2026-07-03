#!/usr/bin/env python3
"""
Bangla normalization pass using bnunicodenormalizer.

Reads from deduped/, writes to cleaned/.
Run after dedup (01b, 01c) to apply full Bangla-specific normalization.

Usage:
    python scripts/pipeline/01d_bn_normalize.py                                    # both files
    python scripts/pipeline/01d_bn_normalize.py --files deduped/bangla_deduped.jsonl
    python scripts/pipeline/01d_bn_normalize.py --input deduped/sangraha_deduped.jsonl --output cleaned/sangraha.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from bnunicodenormalizer import Normalizer
except ImportError:
    print("pip install bnunicodenormalizer")
    sys.exit(1)

DEDUPED_DIR = Path("saved/data/deduped")
CLEANED_DIR = Path("saved/data/cleaned")

DEFAULT_FILES = [
    "saved/data/deduped/bangla_deduped.jsonl",
    "saved/data/deduped/sangraha_deduped.jsonl",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply bnunicodenormalizer to Bangla text in deduped JSONL files."
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Deduped JSONL files to normalize (default: all in deduped/)",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single input file (alternative to --files)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Single output file (used with --input)",
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


def process_file(input_path: Path, output_path: Path, norm: Normalizer,
                 prefix: str, dry_run: bool) -> dict:
    """Read input, normalize, write to output."""
    stats = {"read": 0, "normalized": 0, "skipped": 0, "bytes": input_path.stat().st_size}

    if dry_run:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                stats["read"] += 1
                doc = json.loads(line)
                if doc.get("language_region", "").startswith("BD"):
                    stats["normalized"] += 1
                else:
                    stats["skipped"] += 1
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stats["read"] += 1
            doc = json.loads(line)

            if not doc.get("language_region", "").startswith("BD"):
                stats["skipped"] += 1
                fout.write(line)
                continue

            doc["text"] = normalize_words(norm, doc["text"], prefix)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            stats["normalized"] += 1

    return stats


def main():
    args = parse_args()
    norm = Normalizer()

    # Build file list: (input_path, output_path) pairs
    if args.input and args.output:
        files = [(Path(args.input), Path(args.output))]
    elif args.files:
        files = []
        for f in args.files:
            p = Path(f)
            out = CLEANED_DIR / p.name
            files.append((p, out))
    else:
        files = []
        for f in DEFAULT_FILES:
            p = Path(f)
            if p.exists():
                out = CLEANED_DIR / p.name
                files.append((p, out))

    if not files:
        print("No files to normalize.")
        return

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    total = {"read": 0, "normalized": 0, "skipped": 0, "bytes": 0}
    start = time.time()

    for input_path, output_path in files:
        if not input_path.exists():
            print(f"  skipping {input_path} (not found)")
            continue

        print(f"  {input_path.name} → {output_path.name}")
        s = process_file(input_path, output_path, norm, args.prefix, args.dry_run)
        mb = s["bytes"] / 1e6
        print(f"    {s['normalized']:,} normalized, "
              f"{s['skipped']:,} skipped ({mb:.0f} MB)")

        for k in total:
            total[k] += s[k]

    elapsed = time.time() - start
    mb = total["bytes"] / 1e6
    print(f"\nTotal: {total['normalized']:,} Bangla docs normalized, "
          f"{total['skipped']:,} non-Bangla skipped ({mb:.0f} MB)")
    print(f"Time: {elapsed:.1f}s")
    if args.dry_run:
        print("*** DRY RUN — no output written ***")


if __name__ == "__main__":
    main()
