"""
Download NLLB (ben_Beng-eng_Latn) → JSONL

Output (v3 schema — two lines per pair):
  {"text": "<|task_translate_bn_en|><|lang_bn|>...<|lang_en|>...",
   "source": "nllb", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}
  {"text": "<|task_translate_en_bn|><|lang_en|>...<|lang_bn|>...",
   "source": "nllb", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}

Usage:
  pip install datasets tqdm
  python scripts/downloaders/03b_download_nllb.py

Edit SETTINGS below before running.
"""

from pathlib import Path
from tqdm import tqdm

from _common import (
    RAW_DIR, LANG_BN, LANG_EN, TASK_BN_EN, TASK_EN_BN,
    count_lines, write_doc, normalize_text, wc,
    length_ok, ratio_ok, is_dup_nmt,
)

# ── Settings ──────────────────────────────────────────────────────────────────
LASER_THRESHOLD = 1.06
MAX_PAIRS       = None
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT = RAW_DIR / "nllb.jsonl"
SOURCE = "nllb"
SOURCE_TYPE = "parallel_bn_en"
LANGUAGE_REGION = "BN_EN_parallel"


def download_nllb():
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)
    existing_pairs = existing // 2

    print("── Downloading NLLB ben_Beng-eng_Latn ──")
    print(f"   Streaming from HuggingFace (no 400GB download)")
    print(f"   LASER threshold : {LASER_THRESHOLD}")
    print(f"   Cap             : {MAX_PAIRS or 'none'}")
    print(f"   Existing        : {existing_pairs:,} pairs")

    ds = load_dataset(
        "allenai/nllb",
        "ben_Beng-eng_Latn",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    kept = dropped = 0

    with open(OUTPUT, "a") as f:
        skip = existing_pairs
        for row in tqdm(ds, desc="NLLB", unit=" pairs"):
            if MAX_PAIRS and kept >= MAX_PAIRS:
                print(f"\n   Cap reached at {MAX_PAIRS:,} pairs.")
                break

            if skip > 0:
                skip -= 1
                continue

            t     = row["translation"]
            score = row.get("laser_score", 0.0)
            bn    = normalize_text(t.get("ben_Beng", "").strip())
            en    = normalize_text(t.get("eng_Latn", "").strip())

            if not bn or not en:
                dropped += 1
                continue
            if score < LASER_THRESHOLD:
                dropped += 1
                continue
            if not length_ok(bn, en) or not ratio_ok(bn, en):
                dropped += 1
                continue
            if is_dup_nmt(bn):
                dropped += 1
                continue

            n = wc(bn) + wc(en)
            write_doc(f, f"{TASK_BN_EN}{LANG_BN}{bn}{LANG_EN}{en}",
                      SOURCE, SOURCE_TYPE, LANGUAGE_REGION, n)
            write_doc(f, f"{TASK_EN_BN}{LANG_EN}{en}{LANG_BN}{bn}",
                      SOURCE, SOURCE_TYPE, LANGUAGE_REGION, n)
            kept += 1

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\nDone.")
    print(f"  Kept           : {kept:>12,}")
    print(f"  Dropped        : {dropped:>12,}")
    print(f"  Output         : {OUTPUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_nllb()
