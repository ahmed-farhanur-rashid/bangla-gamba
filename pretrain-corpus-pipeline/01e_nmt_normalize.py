#!/usr/bin/env python3
"""
Bangla NMT normalization pass using bn-normalize-rs (Rust).

Reads from a JSONL file with translation pairs, normalizes ONLY the Bangla portions,
and leaves the English (or other language) portions intact.
If the Bangla portion contains invalid words that fail normalization, the ENTIRE
translation pair document is dropped.

Usage:
    python pretrain-corpus-pipeline/01e_nmt_normalize.py
    python pretrain-corpus-pipeline/01e_nmt_normalize.py --input saved/data/deduped/nmt_deduped.jsonl --output saved/data/cleaned/nmt.jsonl
    python pretrain-corpus-pipeline/01e_nmt_normalize.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

try:
    import bn_normalize_rs
except ImportError:
    print("Install Rust normalizer: cd bn-normalizer-rs && pip install -e .")
    sys.exit(1)

LOG_DIR = Path("saved/logs")

DEFAULT_INPUT = Path("saved/data/deduped/nmt_deduped.jsonl")
DEFAULT_OUTPUT = Path("saved/data/cleaned/nmt.jsonl")


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply bn-normalize-rs to Bangla text inside NMT JSONL files."
    )
    p.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Input JSONL file (default: saved/data/deduped/nmt_deduped.jsonl)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL file (default: saved/data/cleaned/nmt.jsonl)",
    )
    p.add_argument(
        "--none-policy",
        default="drop_and_collect",
        choices=["drop", "keep_original", "error", "collect", "drop_and_collect"],
        help="What to do when a Bangla word normalizes to None (default: drop_and_collect)",
    )
    p.add_argument(
        "--allow-english",
        action="store_true",
        default=True,
        help="Treat English characters as valid in Bangla word normalization (default: True)",
    )
    p.add_argument(
        "--no-allow-english",
        action="store_true",
        help="Disable allow_english in Bangla word normalization",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    return p.parse_args()


def process_nmt_text(text: str, none_policy: str, allow_english: bool):
    """
    Parses the NMT text string, normalizes the <|lang_bn|> segments,
    and returns the combined processed text along with any failed tokens.
    """
    pieces = re.split(r'(<\|[^>]+\|>)', text)
    out_pieces = []
    failed_tokens_total = []
    
    current_lang = None
    
    for piece in pieces:
        if piece.startswith("<|") and piece.endswith("|>"):
            current_lang = piece
            out_pieces.append(piece)
        elif piece:
            if current_lang == "<|lang_bn|>":
                result = bn_normalize_rs.normalize_sentence(piece, none_policy, allow_english)
                if none_policy in ("collect", "drop_and_collect") and isinstance(result, tuple):
                    norm_text, failed_tokens = result
                    out_pieces.append(norm_text)
                    failed_tokens_total.extend(failed_tokens)
                else:
                    out_pieces.append(result)
            else:
                out_pieces.append(piece)
                
    return "".join(out_pieces), failed_tokens_total


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def process_file(input_path: Path, output_path: Path, none_policy: str,
                 allow_english: bool, dry_run: bool) -> dict:
    """Read input, normalize NMT format, write to output and drop failed docs."""
    stats = {"read": 0, "normalized": 0, "dropped": 0, "bytes": input_path.stat().st_size}
    total_failed = 0
    failed_log_path = None
    failed_log = None

    total = count_lines(input_path)

    if dry_run:
        with open(input_path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  {input_path.stem} (dry run)", total=total, unit="docs", unit_scale=True):
                stats["read"] += 1
                doc = json.loads(line)
                
                _, failed_tokens = process_nmt_text(doc.get("text", ""), none_policy, allow_english)
                if failed_tokens:
                    stats["dropped"] += 1
                else:
                    stats["normalized"] += 1
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if none_policy in ("collect", "drop_and_collect"):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        failed_log_path = LOG_DIR / f"{input_path.stem}_nmt_norm_failures.jsonl"
        failed_log = open(failed_log_path, "w", encoding="utf-8")

    try:
        with open(input_path, encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"  {input_path.stem}", total=total, unit="docs", unit_scale=True):
                stats["read"] += 1
                doc = json.loads(line)

                processed_text, failed_tokens = process_nmt_text(doc.get("text", ""), none_policy, allow_english)

                if none_policy in ("collect", "drop_and_collect") and failed_tokens:
                    total_failed += len(failed_tokens)
                    stats["dropped"] += 1
                    
                    for pos, token in failed_tokens:
                        failed_log.write(json.dumps({
                            "source": doc.get("source", ""),
                            "position": pos,
                            "token": token,
                            "original_text": doc.get("text", "")
                        }, ensure_ascii=False) + "\n")
                    # Drop the whole document
                    continue

                doc["text"] = processed_text
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

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    print(f"  {input_path.name} → {output_path.name}")
    start = time.time()
    
    s = process_file(input_path, output_path, none_policy, allow_english, args.dry_run)
    
    mb = s["bytes"] / 1e6
    failed = s.get("failed_tokens", 0)
    
    print(f"    {s['normalized']:,} valid docs kept, "
          f"{s['dropped']:,} docs dropped due to bad Bangla ({mb:.0f} MB)")
    if failed:
        print(f"    {failed:,} bad words triggered doc drops → {s.get('failed_log', '')}")

    elapsed = time.time() - start
    print(f"\nTotal: {s['normalized']:,} valid NMT docs processed, "
          f"{s['dropped']:,} dropped.")
    if failed:
        print(f"Failed tokens: {failed:,} (see saved/logs/)")
    print(f"Time: {elapsed:.1f}s")
    if args.dry_run:
        print("*** DRY RUN — no output written ***")


if __name__ == "__main__":
    main()
