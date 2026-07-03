"""
Cross-source monolingual dedup — TituLLM ∩ Wiki Bangla ∩ Sangraha Verified.

Exact dedup via SHA-256 on text content.
Catches documents that appear in multiple sources.
Also applies full Bangla normalization (bnunicodenormalizer) on write,
so the separate 01d_bn_normalize.py step is no longer needed.

Priority order: wiki_bangla > titullm > sangraha_verified
(First source wins on conflict; later duplicates are dropped.)

Input:  saved/data/raw/wiki_bangla.jsonl
        saved/data/raw/titullm_cc.jsonl
        saved/data/raw/sangraha_verified_bn.jsonl
Output: saved/data/cleaned/bangla.jsonl

Usage:
  python scripts/pipeline/01b_dedup_mono_bn.py
  python scripts/pipeline/01b_dedup_mono_bn.py --delete-raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

try:
    from bnunicodenormalizer import Normalizer
except ImportError:
    print("pip install bnunicodenormalizer")
    sys.exit(1)

RAW_DIR = Path("saved/data/raw")
CLEANED_DIR = Path("saved/data/cleaned")
OUTPUT = CLEANED_DIR / "bangla.jsonl"

SOURCES = [
    ("wiki_bangla", RAW_DIR / "wiki_bangla.jsonl"),
    ("titullm", RAW_DIR / "titullm_cc.jsonl"),
    ("sangraha_verified", RAW_DIR / "sangraha_verified_bn.jsonl"),
]


def normalize_for_hash(text: str) -> bytes:
    """Normalize text before hashing so trivial formatting/encoding
    differences (whitespace, Unicode form) don't defeat exact dedup."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("utf-8")


LANG_BN = "<|lang_bn|>"


def normalize_words(norm: Normalizer, text: str) -> str:
    """Strip <|lang_bn|> prefix, normalize each word, re-add prefix."""
    stripped = text
    had_prefix = stripped.startswith(LANG_BN)
    if had_prefix:
        stripped = stripped[len(LANG_BN):].lstrip()

    words = stripped.split()
    normalized = []
    for w in words:
        result = norm(w)
        if isinstance(result, dict):
            n = result.get("normalized") or result.get("text") or w
            normalized.append(n)
        else:
            normalized.append(result if result else w)

    out = " ".join(normalized)
    if had_prefix:
        out = LANG_BN + " " + out
    return out


def main():
    parser = argparse.ArgumentParser(description="Cross-source monolingual dedup.")
    parser.add_argument("--delete-raw", action="store_true",
                        help="Delete Bangla raw files after dedup.")
    args = parser.parse_args()

    norm = Normalizer()
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Check inputs exist
    existing = [(name, path) for name, path in SOURCES if path.exists()]

    if not existing:
        print("[dedup_mono] No Bangla raw files found.")
        return

    print(f"[dedup_mono] Input files:")
    for name, path in existing:
        print(f"             {name}: {path}")

    # Count total lines
    total = 0
    for _, path in existing:
        with open(path, "rb") as f:
            for _ in f:
                total += 1

    # Exact dedup via SHA-256
    seen_hashes: dict[bytes, str] = {}   # hash -> source_name that kept it
    kept = 0
    empty_text = 0
    dupes_by_source = {name: 0 for name, _ in existing}      # docs dropped, by their own source
    dupes_vs_source = {name: 0 for name, _ in existing}      # docs dropped, by the source that "won"

    with open(OUTPUT, "w") as fout:
        with tqdm(total=total, desc="Mono dedup", unit="docs", unit_scale=True) as bar:
            for source_name, path in existing:
                with open(path, "r") as f:
                    for line in f:
                        bar.update(1)
                        try:
                            doc = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        text = doc.get("text", "")
                        hash_bytes = normalize_for_hash(text)
                        if not hash_bytes:
                            empty_text += 1
                            continue

                        h = hashlib.sha256(hash_bytes).digest()

                        if h in seen_hashes:
                            dupes_by_source[source_name] += 1
                            dupes_vs_source[seen_hashes[h]] += 1
                            continue
                        seen_hashes[h] = source_name

                        doc["text"] = normalize_words(norm, text)
                        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        kept += 1

    # Stats
    in_size = sum(p.stat().st_size for _, p in existing) / (1024 ** 3)
    out_size = OUTPUT.stat().st_size / (1024 ** 3)

    total_dupes = sum(dupes_by_source.values())

    print(f"\n{'=' * 50}")
    print(f"=== MONO DEDUP COMPLETE ===")
    print(f"  Input docs:     {total:,}")
    print(f"  Empty text:     {empty_text:,}")
    print(f"  Duplicates:     {total_dupes:,}")
    print(f"  Kept:           {kept:,}")
    print(f"  Output:         {out_size:.1f} GB  →  {OUTPUT}")
    print(f"  Input:          {in_size:.1f} GB")
    print(f"  --- Dupes dropped, by their own source ---")
    for name, _ in existing:
        print(f"    {name}: {dupes_by_source[name]:,} docs dropped as duplicates")
    print(f"  --- Duplicates absorbed by the surviving copy's source ---")
    for name, _ in existing:
        print(f"    {name}: {dupes_vs_source[name]:,} later dupes matched a doc kept from here")
    print(f"{'=' * 50}")

    if args.delete_raw:
        for _, path in existing:
            if path.exists():
                path.unlink()
                print(f"[dedup_mono] Deleted {path}")


if __name__ == "__main__":
    main()
