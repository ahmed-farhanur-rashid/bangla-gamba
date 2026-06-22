"""
Download TituLM Common Crawl Bangla corpus.
4.5M docs, streamed. Supports mid-download resumption.

Usage:
  python scripts/download/01a_download_titulm_cc.py
  python scripts/download/01a_download_titulm_cc.py --doc-limit 6000000
  python scripts/download/01a_download_titulm_cc.py --max-docs 5000  # test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import RAW_DIR, count_lines, write_doc, normalize_text, has_min_words


OUTPUT = RAW_DIR / "titulm_cc.jsonl"
SOURCE = "titulm_cc"
SOURCE_TYPE = "web_bangla"
LANGUAGE_REGION = "BD_WB_mix"
DEFAULT_LIMIT = 4_500_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-limit", type=int, default=None,
                        help="Override default doc limit (4.5M).")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs.")
    args = parser.parse_args()

    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    limit = args.doc_limit or args.max_docs or DEFAULT_LIMIT
    existing = count_lines(OUTPUT)

    if existing >= limit:
        print(f"  \u21b7 titulm_cc already complete ({existing:,} docs), skipping")
        return

    ds = load_dataset("hishab/titulm-bangla-corpus", data_dir="common_crawl",
                       split="train", streaming=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="TituLM CC       ", unit="docs", unit_scale=True,
                   initial=existing, total=limit)
        for i, row in enumerate(ds):
            if i < existing:
                continue
            if i >= limit:
                break
            text = normalize_text(row.get("text", ""))
            if has_min_words(text):
                write_doc(f, text, SOURCE, SOURCE_TYPE, LANGUAGE_REGION)
            bar.update(1)
        bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 titulm_cc \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
