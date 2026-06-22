"""
BanglaGamba — Deduplicate Cleaned Corpus
==========================================
Streaming dedup on saved/data/cleaned/corpus_cleaned.jsonl.

  Level 1: Exact dedup via SHA-256  (~400 MB RAM for 12M docs)
  Level 2: Fuzzy dedup via MinHash LSH  (requires ``datasketch``)

Writes corpus_deduped.jsonl alongside the original, then deletes
the original to reclaim disk.

Usage
-----
  pip install datasketch          # one-time
  python scripts/pipeline/02b_dedup.py
  python scripts/pipeline/02b_dedup.py --exact-only
  python scripts/pipeline/02b_dedup.py --threshold 0.85 --num-perm 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

INPUT_PATH  = Path("saved/data/cleaned/corpus_cleaned.jsonl")
OUTPUT_PATH = Path("saved/data/cleaned/corpus_deduped.jsonl")


# ── helpers ──────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _make_minhash(text: str, MinHash, num_perm: int):
    """Build a MinHash from 5-gram shingles of *text*."""
    m = MinHash(num_perm=num_perm)
    words = text.split()
    for i in range(max(len(words) - 4, 1)):
        gram = " ".join(words[i : i + 5])
        m.update(gram.encode("utf-8"))
    return m


# ── main ─────────────────────────────────────────────────────────────────────

def run_dedup(
    exact_only: bool = False,
    threshold: float = 0.80,
    num_perm: int = 128,
    no_delete: bool = False,
):
    if not INPUT_PATH.exists():
        print(f"[dedup] ERROR: {INPUT_PATH} not found.")
        return

    # ── optional datasketch ──────────────────────────────────────────────
    MinHash = MinHashLSH = None
    if not exact_only:
        try:
            from datasketch import MinHash as _MH, MinHashLSH as _LSH
            MinHash, MinHashLSH = _MH, _LSH
        except ImportError:
            print("[dedup] WARNING: datasketch not installed — exact dedup only.")
            print("[dedup]          Install with:  pip install datasketch")
            exact_only = True

    mode = "exact only" if exact_only else f"exact + fuzzy (threshold={threshold}, perm={num_perm})"
    print(f"[dedup] Input:  {INPUT_PATH}")
    print(f"[dedup] Output: {OUTPUT_PATH}")
    print(f"[dedup] Mode:   {mode}")

    # ── count ────────────────────────────────────────────────────────────
    print("[dedup] Counting documents...")
    total = _count_lines(INPUT_PATH)
    print(f"[dedup] Total: {total:,}")

    # ── structures ───────────────────────────────────────────────────────
    seen_hashes: set[bytes] = set()
    lsh = None
    if not exact_only:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    exact_dupes = 0
    fuzzy_dupes = 0
    kept = 0

    # ── stream ───────────────────────────────────────────────────────────
    with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
        for line in tqdm(fin, total=total, desc="Dedup", unit="docs", unit_scale=True):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "")

            # — exact —
            h = hashlib.sha256(text.encode("utf-8")).digest()
            if h in seen_hashes:
                exact_dupes += 1
                continue
            seen_hashes.add(h)

            # — fuzzy —
            if lsh is not None and len(text.split()) >= 5:
                mh = _make_minhash(text, MinHash, num_perm)
                if lsh.query(mh):
                    fuzzy_dupes += 1
                    continue
                try:
                    lsh.insert(str(kept), mh)
                except ValueError:
                    pass  # key collision (shouldn't happen with int keys)

            doc["doc_id"] = kept
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    # ── cleanup ──────────────────────────────────────────────────────────
    in_gb  = INPUT_PATH.stat().st_size / (1024 ** 3)
    out_gb = OUTPUT_PATH.stat().st_size / (1024 ** 3)

    if not no_delete:
        INPUT_PATH.unlink()
        freed_msg = f"  freed {in_gb:.1f} GB"
    else:
        freed_msg = "  (original kept — --no-delete)"

    removed = exact_dupes + fuzzy_dupes
    print(f"\n{'=' * 60}")
    print("=== DEDUPLICATION COMPLETE ===")
    print(f"  Input docs:         {total:,}")
    print(f"  Exact duplicates:   {exact_dupes:,}")
    print(f"  Fuzzy duplicates:   {fuzzy_dupes:,}")
    print(f"  Removed total:      {removed:,}  ({removed / max(total, 1) * 100:.1f}%)")
    print(f"  Kept:               {kept:,}  ({kept / max(total, 1) * 100:.1f}%)")
    print(f"  Output:             {out_gb:.1f} GB  →  {OUTPUT_PATH}")
    print(f"  Original:           {in_gb:.1f} GB{freed_msg}")
    print(f"{'=' * 60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Deduplicate corpus_cleaned.jsonl")
    ap.add_argument("--exact-only", action="store_true",
                    help="Skip fuzzy (MinHash) dedup — exact hash dedup only.")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="MinHash LSH Jaccard threshold (default 0.80).")
    ap.add_argument("--num-perm", type=int, default=128,
                    help="MinHash permutations (default 128).")
    ap.add_argument("--no-delete", action="store_true",
                    help="Keep the original corpus_cleaned.jsonl after dedup.")
    args = ap.parse_args()
    run_dedup(
        exact_only=args.exact_only,
        threshold=args.threshold,
        num_perm=args.num_perm,
        no_delete=args.no_delete,
    )


if __name__ == "__main__":
    main()
