"""
Download NLLB (ben_Beng-eng_Latn) → JSONL

Downloads the raw gzipped TSV directly from Google Cloud Storage
(no HuggingFace datasets loading script needed).

Output (v3 schema — two lines per pair):
  {"text": "<|task_translate_bn_en|><|lang_bn|>...<|lang_en|>...",
   "source": "nllb", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}
  {"text": "<|task_translate_en_bn|><|lang_en|>...<|lang_bn|>...",
   "source": "nllb", "source_type": "parallel_bn_en",
   "language_region": "BN_EN_parallel", "word_count": N}

Usage:
  python pretrain-corpus-pipeline/downloaders/03b_download_nllb.py

Edit SETTINGS below before running.
"""

import gzip
import hashlib
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    RAW_DIR,
    LANG_BN,
    LANG_EN,
    TASK_BN_EN,
    TASK_EN_BN,
    count_lines,
    write_doc,
    normalize_text,
    wc,
    length_ok,
    ratio_ok,
    is_dup_nmt,
)

# ── Settings ──────────────────────────────────────────────────────────────────
LASER_THRESHOLD = 1.06
MAX_PAIRS       = None
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT = RAW_DIR / "nllb.jsonl"
SOURCE = "nllb"
SOURCE_TYPE = "parallel_bn_en"
LANGUAGE_REGION = "BN_EN_parallel"

# AllenAI GCS bucket — raw gzipped TSV for this language pair
GCS_URL = "https://storage.googleapis.com/allennlp-data-bucket/nllb/ben_Beng-eng_Latn.gz"


def download_gzipped(url: str, dest: Path) -> None:
    """Download a gzipped file from URL, showing progress."""
    print(f"   Downloading {url}")
    print(f"   This may take 10–20 minutes depending on connection speed.")

    # Stream download with progress bar
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if not hasattr(_progress, "pbar"):
            _progress.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc="   Downloading",
            )
        _progress.pbar.update(block_size)

    urlretrieve(url, str(dest), reporthook=_progress)
    _progress.pbar.close()


def download_nllb():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = count_lines(OUTPUT)
    existing_pairs = existing // 2

    print("── Downloading NLLB ben_Beng-eng_Latn ──")
    print(f"   LASER threshold : {LASER_THRESHOLD}")
    print(f"   Cap             : {MAX_PAIRS or 'none'}")
    print(f"   Existing        : {existing_pairs:,} pairs")

    # Download the gzipped TSV to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_gzipped(GCS_URL, tmp_path)

        # Parse the gzipped TSV
        print("   Parsing gzipped TSV...")
        kept = dropped = 0

        with gzip.open(tmp_path, "rt", encoding="utf-8") as gz_in, \
             open(OUTPUT, "a") as f:

            skip = existing_pairs

            for line_no, line in enumerate(tqdm(gz_in, desc="   Parsing", unit=" lines")):
                if MAX_PAIRS and kept >= MAX_PAIRS:
                    print(f"\n   Cap reached at {MAX_PAIRS:,} pairs.")
                    break

                if skip > 0:
                    skip -= 1
                    continue

                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    dropped += 1
                    continue

                bn_text = normalize_text(parts[0].strip())
                en_text = normalize_text(parts[1].strip())
                score = float(parts[2]) if parts[2] else 0.0

                if not bn_text or not en_text:
                    dropped += 1
                    continue
                if score < LASER_THRESHOLD:
                    dropped += 1
                    continue
                if not length_ok(bn_text, en_text) or not ratio_ok(bn_text, en_text):
                    dropped += 1
                    continue
                if is_dup_nmt(bn_text):
                    dropped += 1
                    continue

                n = wc(bn_text) + wc(en_text)
                write_doc(
                    f,
                    f"{TASK_BN_EN}{LANG_BN}{bn_text}{LANG_EN}{en_text}",
                    SOURCE,
                    SOURCE_TYPE,
                    LANGUAGE_REGION,
                    n,
                )
                write_doc(
                    f,
                    f"{TASK_EN_BN}{LANG_EN}{en_text}{LANG_BN}{bn_text}",
                    SOURCE,
                    SOURCE_TYPE,
                    LANGUAGE_REGION,
                    n,
                )
                kept += 1

    finally:
        tmp_path.unlink(missing_ok=True)

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\nDone.")
    print(f"  Kept           : {kept:>12,}")
    print(f"  Dropped        : {dropped:>12,}")
    print(f"  Output         : {OUTPUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_nllb()
