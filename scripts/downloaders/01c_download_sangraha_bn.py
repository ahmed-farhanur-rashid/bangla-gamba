"""
Download Sangraha Verified Bengali corpus (AI4Bharat).

Non-Common-Crawl source: human-verified websites + OCR'd PDFs/books +
transcribed speech. Low expected overlap with titullm_cc (CC-derived).

IMPORTANT: config is "verified", split is "ben" for Bengali.
Do NOT use "asm" (Assamese) — same script family, different language,
and datasets.load_dataset will silently succeed on either since both
are valid splits. Double-check before running.

Output (v3 schema):
  {"text": "<|lang_bn|>...", "source": "sangraha_verified", "source_type": "verified_web_ocr",
   "language_region": "BD", "word_count": N}

Usage:
  python scripts/downloaders/01c_download_sangraha_bn.py
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm
from _common import (
    RAW_DIR, LANG_BN, count_lines, write_doc,
    normalize_doc, wc, has_min_words,
)


OUTPUT = RAW_DIR / "sangraha_verified_bn.jsonl"
SOURCE = "sangraha_verified"
SOURCE_TYPE = "verified_web_ocr"
LANGUAGE_REGION = "BD"

# Sangraha structure: config="verified", split="ben" (Bengali).
# Full list of valid splits: https://huggingface.co/datasets/ai4bharat/sangraha
CONFIG = "verified"
SPLIT = "ben"


def main():
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)

    print(f"[sangraha] Loading config={CONFIG!r}, split={SPLIT!r} — "
          f"verify this looks right before leaving this running unattended.")

    ds = load_dataset("ai4bharat/sangraha", CONFIG,
                       split=SPLIT, streaming=True)

    with open(OUTPUT, "a") as f:
        bar = tqdm(desc="Sangraha Verified BN", unit="docs", unit_scale=True,
                   initial=existing)
        written = 0
        skip = existing
        printed_sample = existing > 0  # skip sample print on resume
        for row in ds:
            text = normalize_doc(row.get("text", ""))
            if not has_min_words(text):
                continue
            if not printed_sample:
                print(f"\n[sangraha] SAMPLE DOC (verify this is Bengali, not Assamese):\n"
                      f"{text[:200]}\n")
                printed_sample = True
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
    print(f"  \u2713 sangraha_verified_bn \u2192 {OUTPUT}  ({count:,} docs, {size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
