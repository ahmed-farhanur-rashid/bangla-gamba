#!/usr/bin/env python3
"""
Bangla normalization pass using bn-normalize-rs (Rust).

Reads from deduped/, writes to cleaned/.
Run after dedup (01b, 01c) to apply full Bangla-specific normalization.

Usage:
    python scripts/pipeline/01d_bn_normalize.py                                    # both files
    python scripts/pipeline/01d_bn_normalize.py --files deduped/bangla_deduped.jsonl
    python scripts/pipeline/01d_bn_normalize.py --input deduped/sangraha_deduped.jsonl --output cleaned/sangraha.jsonl
    python scripts/pipeline/01d_bn_normalize.py --none-policy drop                 # drop failed words
    python scripts/pipeline/01d_bn_normalize.py --dry-run                          # stats only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

try:
    import bn_normalize_rs
except ImportError:
    print("Install Rust normalizer: cd bn-normalizer-rs && pip install -e .")
    sys.exit(1)

DEDUPED_DIR = Path("saved/data/deduped")
CLEANED_DIR = Path("saved/data/cleaned")
LOG_DIR = Path("saved/data/logs")

DEFAULT_FILES = [
    "saved/data/deduped/bangla_deduped.jsonl",
    "saved/data/deduped/sangraha_deduped.jsonl",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply bn-normalize-rs (Rust) to Bangla text in deduped JSONL files."
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
        "--none-policy",
        default="collect",
        choices=["drop", "keep_original", "error", "collect"],
        help="What to do when a Bangla word normalizes to None (default: collect)",
    )
    p.add_argument(
        "--allow-english",
        action="store_true",
        default=True,
        help="Treat English characters as valid in word normalization (default: True)",
    )
    p.add_argument(
        "--no-allow-english",
        action="store_true",
        help="Disable allow_english",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    return p.parse_args()


def normalize_text(text: str, none_policy: str, allow_english: bool):
    """Normalize a full text string using the Rust normalizer."""
    return bn_normalize_rs.normalize_sentence(text, none_policy, allow_english)


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def process_file(input_path: Path, output_path: Path, none_policy: str,
                 allow_english: bool, dry_run: bool) -> dict:
    """Read input, normalize, write to output."""
    stats = {"read": 0, "normalized": 0, "skipped": 0, "bytes": input_path.stat().st_size}
    total_failed = 0
    failed_log_path = None
    failed_log = None

    total = count_lines(input_path)

    if dry_run:
        with open(input_path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  {input_path.stem} (dry run)", total=total, unit="docs", unit_scale=True):
                stats["read"] += 1
                doc = json.loads(line)
                if doc.get("language_region", "").startswith("BD"):
                    stats["normalized"] += 1
                else:
                    stats["skipped"] += 1
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if none_policy == "collect":
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        failed_log_path = LOG_DIR / f"{input_path.stem}_norm_failures.jsonl"
        failed_log = open(failed_log_path, "w")

    try:
        with open(input_path, encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"  {input_path.stem}", total=total, unit="docs", unit_scale=True):
                stats["read"] += 1
                doc = json.loads(line)

                if not doc.get("language_region", "").startswith("BD"):
                    stats["skipped"] += 1
                    fout.write(line)
                    continue

                result = normalize_text(doc["text"], none_policy, allow_english)

                if none_policy == "collect" and isinstance(result, tuple):
                    doc["text"], failed_tokens = result
                    if failed_tokens:
                        total_failed += len(failed_tokens)
                        for pos, token in failed_tokens:
                            failed_log.write(json.dumps({
                                "doc_id": doc.get("doc_id", ""),
                                "source": doc.get("source", ""),
                                "position": pos,
                                "token": token,
                            }, ensure_ascii=False) + "\n")
                else:
                    doc["text"] = result

                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                stats["normalized"] += 1
    finally:
        if failed_log:
            failed_log.close()

    stats["failed_tokens"] = total_failed
    if failed_log_path and total_failed == 0:
        failed_log_path.unlink(missing_ok=True)
    elif failed_log_path:
        stats["failed_log"] = str(failed_log_path)

    return stats


def main():
    args = parse_args()
    none_policy = args.none_policy
    allow_english = args.allow_english and not args.no_allow_english

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
    total_failed = 0
    start = time.time()

    for input_path, output_path in files:
        if not input_path.exists():
            print(f"  skipping {input_path} (not found)")
            continue

        print(f"  {input_path.name} → {output_path.name}")
        s = process_file(input_path, output_path, none_policy, allow_english, args.dry_run)
        mb = s["bytes"] / 1e6
        failed = s.get("failed_tokens", 0)
        total_failed += failed
        print(f"    {s['normalized']:,} normalized, "
              f"{s['skipped']:,} skipped ({mb:.0f} MB)")
        if failed:
            print(f"    {failed:,} words failed normalization → {s.get('failed_log', '')}")

        for k in ["read", "normalized", "skipped", "bytes"]:
            total[k] += s[k]

    elapsed = time.time() - start
    mb = total["bytes"] / 1e6
    print(f"\nTotal: {total['normalized']:,} Bangla docs normalized, "
          f"{total['skipped']:,} non-Bangla skipped ({mb:.0f} MB)")
    if total_failed:
        print(f"Failed tokens: {total_failed:,} (see saved/data/logs/)")
    print(f"Time: {elapsed:.1f}s")
    if args.dry_run:
        print("*** DRY RUN — no output written ***")


if __name__ == "__main__":
    main()
