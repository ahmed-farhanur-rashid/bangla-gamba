"""
Download BanglaNMT parallel Bangla-English pairs.
2.75M pairs, loaded fully. Writes as "bn |SEP| en" per line.

Usage:
  python scripts/download/01d_download_banglanmt.py
  python scripts/download/01d_download_banglanmt.py --max-docs 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import RAW_DIR, count_lines, write_doc, normalize_text, has_min_words


OUTPUT = RAW_DIR / "banglanmt_parallel.jsonl"
SOURCE = "banglanmt"
SOURCE_TYPE = "parallel_bn_en"
LANGUAGE_REGION = "EN_in_BN_context"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs.")
    args = parser.parse_args()

    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)

    ds = load_dataset("csebuetnlp/BanglaNMT", split="train", streaming=False,
                       trust_remote_code=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="BanglaNMT       ", unit="docs", unit_scale=True,
                   initial=existing, total=len(ds))
        for i, row in enumerate(ds):
            if i < existing:
                continue
            if args.max_docs and i >= args.max_docs:
                break

            if "translation" in row:
                bn_text = row["translation"].get("bn", "")
                en_text = row["translation"].get("en", "")
            else:
                bn_text = row.get("bn", "")
                en_text = row.get("en", "")

            if len(bn_text.split()) < 5 or len(en_text.split()) < 5:
                bar.update(1)
                continue

            text = normalize_text(f"{bn_text} |SEP| {en_text}")
            if not has_min_words(text, min_words=10):
                bar.update(1)
                continue

            write_doc(f, text, SOURCE, SOURCE_TYPE, LANGUAGE_REGION)
            bar.update(1)
        bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 banglanmt \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
