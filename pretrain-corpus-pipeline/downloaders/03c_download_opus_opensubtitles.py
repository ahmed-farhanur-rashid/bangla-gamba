"""
Download OPUS OpenSubtitles parallel Bangla-English pairs.
Downloads using the HuggingFace datasets library (opus100 bn-en split).

Output (v3 schema — two lines per pair):
  {"text": "<|task_translate_bn_en|><|lang_bn|>...<|lang_en|>...",
   "source": "opus_opensubtitles", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}
  {"text": "<|task_translate_en_bn|><|lang_en|>...<|lang_bn|>...",
   "source": "opus_opensubtitles", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}

Usage:
  python pretrain-corpus-pipeline/downloaders/03c_download_opus_opensubtitles.py
  python pretrain-corpus-pipeline/downloaders/03c_download_opus_opensubtitles.py --max-docs 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from _common import (
    RAW_DIR, LANG_BN, LANG_EN, TASK_BN_EN, TASK_EN_BN,
    count_lines, write_doc, normalize_text, wc,
    length_ok, ratio_ok, is_dup_nmt,
)

OUTPUT = RAW_DIR / "opus_opensubtitles.jsonl"
SOURCE = "opus_opensubtitles"
SOURCE_TYPE = "parallel_bn_en"
LANGUAGE_REGION = "BN_EN_parallel"
TARGET_DOCS = 50000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=TARGET_DOCS,
                        help="Number of pairs to download (default: 50000).")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)
    existing_pairs = existing // 2
    if existing_pairs >= args.max_docs:
        print(f"  \u21b7 {SOURCE} already complete ({existing_pairs:,} pairs), skipping")
        return

    print(f"[{SOURCE}] Loading opus100 bn-en dataset from HF Hub...")
    ds = load_dataset("opus100", "bn-en", split="train", streaming=True)

    written = existing_pairs
    kept = dropped = 0

    bar = tqdm(desc="OPUS Subs       ", total=args.max_docs, unit="pairs")
    if written > 0:
        bar.update(written)

    skip = existing_pairs
    with open(OUTPUT, "a", encoding="utf-8") as fout:
        for row in ds:
            if skip > 0:
                skip -= 1
                continue
            if written >= args.max_docs:
                break
            
            translation = row.get("translation", {})
            bn_text = normalize_text(translation.get("bn", ""))
            en_text = normalize_text(translation.get("en", ""))

            if not length_ok(bn_text, en_text) or not ratio_ok(bn_text, en_text) or is_dup_nmt(bn_text):
                dropped += 1
                continue

            n = wc(bn_text) + wc(en_text)
            
            # Flip them to get dual en->bn and bn->en datasets
            write_doc(fout, f"{TASK_BN_EN}{LANG_BN}{bn_text}{LANG_EN}{en_text}",
                      SOURCE, SOURCE_TYPE, LANGUAGE_REGION, n)
            write_doc(fout, f"{TASK_EN_BN}{LANG_EN}{en_text}{LANG_BN}{bn_text}",
                      SOURCE, SOURCE_TYPE, LANGUAGE_REGION, n)
            
            kept += 1
            written += 1
            bar.update(1)

    bar.close()

    print(f"[{SOURCE}] kept={kept:,} dropped={dropped:,} "
          f"({dropped / max(kept + dropped, 1) * 100:.1f}% removed)")

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 {SOURCE} \u2192 {OUTPUT}  ({count:,} lines, {size_gb:.4f} GB)")

    # Avoid PyArrow/datasets PyGILState_Release crash on exit
    import os
    os._exit(0)

if __name__ == "__main__":
    main()
