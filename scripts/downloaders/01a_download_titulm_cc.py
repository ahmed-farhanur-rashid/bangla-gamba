"""
Download TituLM Common Crawl Bangla corpus.
4.5M docs, streamed. Supports mid-download resumption.

Output (v3 schema):
  {"text": "<|lang_bn|>...", "source": "titullm", "source_type": "web_bangla",
   "language_region": "BD", "word_count": N}

Usage:
  python scripts/downloaders/01a_download_titulm_cc.py
  python scripts/downloaders/01a_download_titulm_cc.py --doc-limit 6000000
  python scripts/downloaders/01a_download_titulm_cc.py --max-docs 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from _common import (
    RAW_DIR, LANG_BN, count_lines, write_doc,
    normalize_doc, wc, has_min_words,
)


OUTPUT = RAW_DIR / "titullm.jsonl"
SOURCE = "titullm"
SOURCE_TYPE = "web_bangla"
LANGUAGE_REGION = "BD"
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
        print(f"  \u21b7 titullm already complete ({existing:,} docs), skipping")
        return

    ds = load_dataset("hishab/titulm-bangla-corpus", data_dir="common_crawl",
                       split="train", streaming=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="TituLM CC       ", unit="docs", unit_scale=True,
                   initial=existing, total=limit)
        written = 0
        skip = existing
        for row in ds:
            if written + existing >= limit:
                break
            text = normalize_doc(row.get("text", ""))
            if not has_min_words(text):
                continue
            if skip > 0:
                skip -= 1
                bar.update(1)
                continue
            n = wc(text)
            write_doc(f, f"{LANG_BN}{text}", SOURCE, SOURCE_TYPE,
                      LANGUAGE_REGION, n)
            written += 1
            bar.update(1)
        bar.close()

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 titullm \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
