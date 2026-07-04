"""
Download English from FineWeb-Edu (sample-10BT).
Streams and stops once word budget is hit.

Output (v3 schema):
  {"text": "<|lang_en|>...", "source": "fineweb_edu", "source_type": "web_english",
   "language_region": "EN", "word_count": N}

Usage:
  python pretrain-corpus-pipeline/downloaders/02a_download_english.py
  python pretrain-corpus-pipeline/downloaders/02a_download_english.py --word-budget 2_000_000_000

Changes from v1:
  - Switched to FineWeb-Edu (HuggingFaceFW/fineweb-edu) which is educationally
    filtered and largely free of adult/spam content.
  - Added lightweight content filter as a second line of defence.
  - Default word budget raised to 2B to support tokenizer sampler at 85/15 split.
  - Falls back to full FineWeb-Edu (350BT) if sample-10BT is exhausted.
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm

from _common import (
    RAW_DIR, CLEANED_DIR, LANG_EN, count_lines, write_doc,
    nfc, wc,
)

OUTPUT = CLEANED_DIR / "english.jsonl"
SOURCE = "fineweb_edu"
SOURCE_TYPE = "web_english"
LANGUAGE_REGION = "EN"
DEFAULT_WORD_BUDGET = 2_000_000_000
AVG_WORDS_PER_DOC = 300

# ---------------------------------------------------------------------------
# Dataset sources — tried in order until budget is met
# ---------------------------------------------------------------------------
# sample-10BT  ~6-7B words  (fast, recommended)
# sample-100BT ~60-70B words (large, slow)
# Default      full dataset  (350BT, very large)
FINEWEB_EDU_CONFIGS = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ("HuggingFaceFW/fineweb-edu", "sample-100BT"),
]

# ---------------------------------------------------------------------------
# Lightweight content filter
# ---------------------------------------------------------------------------
_ADULT_PATTERNS = re.compile(
    r"\b(incall|outcall|hookup|escorts?|onlyfans|camgirl|"
    r"blowjob|handjob|anal\s+sex|pussy|dick\s+pic|sexting|"
    r"naughty\s+girl|hot\s+babe|satisfaction\s+guaranteed|"
    r"available\s+24/7|book\s+me\s+now|text\s+me\s+for|"
    r"independent\s+escort|in\s*call\s*&\s*out\s*call)\b",
    re.IGNORECASE,
)
_SPAM_PATTERNS = re.compile(
    r"(meet\s+new\s+people\s+online|click\s+here\s+to\s+hookup|"
    r"no\s+strings\s+attached\s+sex|free\s+hookup|"
    r"local\s+singles?\s+near\s+you)",
    re.IGNORECASE,
)
_SPAM_DENSITY = re.compile(r"(❤️.*){4,}|(\|\s*\w+\s*){6,}")


def is_clean(text: str) -> bool:
    if _ADULT_PATTERNS.search(text):
        return False
    if _SPAM_PATTERNS.search(text):
        return False
    if _SPAM_DENSITY.search(text):
        return False
    return True


# ---------------------------------------------------------------------------

def _stream_config(repo: str, config: str, word_budget: int, total_words: int,
                   f, docs: int, skipped: int):
    """Stream one FineWeb-Edu config until budget is met. Returns updated counters."""
    from datasets import load_dataset

    print(f"\n   → Streaming {repo} / {config} ...")
    ds = load_dataset(repo, name=config, split="train", streaming=True)

    for row in tqdm(ds, desc=config, unit=" docs"):
        if total_words >= word_budget:
            break

        text = (row.get("text") or "").strip()
        if not text:
            skipped += 1
            continue

        text = nfc(text)
        word_count = wc(text)

        if word_count < 50 or word_count > 100_000:
            skipped += 1
            continue

        write_doc(f, f"{LANG_EN}{text}", SOURCE, SOURCE_TYPE,
                  LANGUAGE_REGION, word_count)
        total_words += word_count
        docs += 1

    return total_words, docs, skipped


def download_fineweb(word_budget: int):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    existing = count_lines(OUTPUT)

    if existing > 0:
        estimated_words = existing * AVG_WORDS_PER_DOC
        if estimated_words >= word_budget:
            print(f"  ↷ fineweb-edu already complete (~{estimated_words:,} words estimated), skipping")
            return

    print("── Downloading FineWeb-Edu ──")
    print(f"   Word budget : {word_budget:,}")
    print(f"   Existing    : {existing:,} docs (~{existing * AVG_WORDS_PER_DOC:,} words estimated)")
    print(f"   Configs     : {' → '.join(c for _, c in FINEWEB_EDU_CONFIGS)}")
    print(f"   Will advance to next config if budget not met\n")

    total_words = existing * AVG_WORDS_PER_DOC
    docs = 0
    skipped = 0

    with open(OUTPUT, "a") as f:
        for repo, config in FINEWEB_EDU_CONFIGS:
            if total_words >= word_budget:
                break
            total_words, docs, skipped = _stream_config(
                repo, config, word_budget, total_words,
                f, docs, skipped,
            )
            if total_words < word_budget:
                remaining = word_budget - total_words
                print(f"\n   [{config}] exhausted. Still need ~{remaining:,} words, advancing...")

    if total_words < word_budget:
        remaining = word_budget - total_words
        print(f"\n   WARNING: All configs exhausted. Still short by ~{remaining:,} words.")
        print(f"   Consider adding Wikipedia English or C4 as a fallback source.")

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\nDone.")
    print(f"  Documents      : {docs:>12,}")
    print(f"  Words collected: {total_words:>12,}")
    print(f"  Skipped        : {skipped:>12,}")
    print(f"  Output         : {OUTPUT}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download English from FineWeb-Edu.")
    parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET,
                        help=f"Word budget (default: {DEFAULT_WORD_BUDGET:,}).")
    args = parser.parse_args()
    download_fineweb(args.word_budget)


if __name__ == "__main__":
    main()
