"""
Download BanglaNMT parallel Bangla-English pairs.
Downloads the tar.bz2 directly from HF Hub, extracts train.jsonl.

Output (v3 schema — two lines per pair):
  {"text": "<|task_translate_bn_en|><|lang_bn|>...<|lang_en|>...",
   "source": "banglanmt", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}
  {"text": "<|task_translate_en_bn|><|lang_en|>...<|lang_bn|>...",
   "source": "banglanmt", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}

Usage:
  python scripts/downloaders/03a_download_banglanmt.py
  python scripts/downloaders/03a_download_banglanmt.py --max-docs 5000
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from tqdm import tqdm
from _common import (
    RAW_DIR, LANG_BN, LANG_EN, TASK_BN_EN, TASK_EN_BN,
    count_lines, write_doc, normalize_text, wc,
    length_ok, ratio_ok, is_dup_nmt,
)


OUTPUT = RAW_DIR / "banglanmt.jsonl"
SOURCE = "banglanmt"
SOURCE_TYPE = "parallel_bn_en"
LANGUAGE_REGION = "BN_EN_parallel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Test mode: download at most N docs (pairs).")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)
    existing_pairs = existing // 2
    if args.max_docs and existing_pairs >= args.max_docs:
        print(f"  \u21b7 banglanmt already complete ({existing_pairs:,} pairs), skipping")
        return

    print("[banglanmt] Downloading data/BanglaNMT.tar.bz2 from HF Hub...")
    tar_path = hf_hub_download(
        "csebuetnlp/BanglaNMT",
        "data/BanglaNMT.tar.bz2",
        repo_type="dataset",
    )
    print(f"[banglanmt] Downloaded to cache: {tar_path}")

    written = existing_pairs
    kept = dropped = 0

    with tarfile.open(tar_path, "r:bz2") as tar:
        f = tar.extractfile("BanglaNMT/train.jsonl")

        bar = tqdm(desc="BanglaNMT       ", unit="pairs", unit_scale=True)

        skip = existing_pairs
        with open(OUTPUT, "a") as fout:
            for line in f:
                if skip > 0:
                    skip -= 1
                    bar.update(1)
                    continue
                if args.max_docs and written >= args.max_docs:
                    break

                row = json.loads(line)
                bn_text = normalize_text(row.get("bn", ""))
                en_text = normalize_text(row.get("en", ""))

                if not length_ok(bn_text, en_text) or not ratio_ok(bn_text, en_text) or is_dup_nmt(bn_text):
                    dropped += 1
                    bar.update(1)
                    continue

                n = wc(bn_text) + wc(en_text)
                write_doc(fout, f"{TASK_BN_EN}{LANG_BN}{bn_text}{LANG_EN}{en_text}",
                          SOURCE, SOURCE_TYPE, LANGUAGE_REGION, n)
                write_doc(fout, f"{TASK_EN_BN}{LANG_EN}{en_text}{LANG_BN}{bn_text}",
                          SOURCE, SOURCE_TYPE, LANGUAGE_REGION, n)
                kept += 1
                written += 1
                bar.update(1)

        bar.close()

    print(f"[banglanmt] kept={kept:,} dropped={dropped:,} "
          f"({dropped / max(kept + dropped, 1) * 100:.1f}% removed)")

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)
    count = count_lines(OUTPUT)
    print(f"  \u2713 banglanmt \u2192 {OUTPUT}  ({count:,} lines, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
