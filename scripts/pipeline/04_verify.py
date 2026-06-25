"""
Verify pretokenized shards before training.

Read-only sanity checks on saved/data/pretokenized/{bangla,english,nmt}/train/.

Usage:
  python scripts/pipeline/05_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PRETOKENIZED_DIR = Path("saved/data/pretokenized")
HF_TOKENIZER_DIR = Path("saved/tokenizer/hf")

SEQ_LEN = 2048
VOCAB_SIZE = 48_000

passed = 0
failed = 0


def check(name: str, ok: bool, msg: str = ""):
    global passed, failed
    if ok:
        print(f"  \u2713 {name}")
        passed += 1
    else:
        print(f"  \u2717 {name}: {msg}")
        failed += 1


def verify_source(source_type: str):
    train_dir = PRETOKENIZED_DIR / source_type / "train"
    print(f"\n── {source_type} ──")

    # 1. Shards exist
    train_shards = sorted(train_dir.glob("shard_*.npy")) if train_dir.exists() else []
    check(f"{source_type} train shards exist", len(train_shards) > 0,
          f"Found {len(train_shards)} shards in {train_dir}")

    if not train_shards:
        return

    print(f"     Shards: {len(train_shards)}")

    # 2. Shape check — first and last shard
    for shard_path in [train_shards[0], train_shards[-1]]:
        arr = np.load(shard_path)
        check(f"Shape {shard_path.name} == (N, {SEQ_LEN})",
              arr.ndim == 2 and arr.shape[1] == SEQ_LEN,
              f"Got {arr.shape}")
        check(f"dtype {shard_path.name} == uint16",
              arr.dtype == np.uint16,
              f"Got {arr.dtype}")

    # 3. Token range — first shard
    first_arr = np.load(train_shards[0])
    token_min = int(first_arr.min())
    token_max = int(first_arr.max())
    check(f"Token range in [0, {VOCAB_SIZE})",
          token_min >= 0 and token_max < VOCAB_SIZE,
          f"min={token_min}, max={token_max}")

    # 4. No all-zero rows — first shard
    zero_rows = int(np.all(first_arr == 0, axis=1).sum())
    total_rows = first_arr.shape[0]
    zero_pct = zero_rows / max(total_rows, 1) * 100
    check("All-zero rows < 1%",
          zero_pct < 1.0,
          f"{zero_rows}/{total_rows} rows ({zero_pct:.1f}%)")

    # 5. Token count
    total_tokens = sum(
        np.load(p).shape[0] * np.load(p).shape[1] for p in train_shards[:3]
    )
    print(f"     Sample tokens (first 3 shards): {total_tokens:,}")


def main():
    print("=" * 60)
    print("  BanglaGamba — Shard Verification")
    print("=" * 60)

    source_types = ["bangla", "english", "nmt"]
    existing_sources = []

    for st in source_types:
        train_dir = PRETOKENIZED_DIR / st / "train"
        if train_dir.exists() and any(train_dir.glob("shard_*.npy")):
            existing_sources.append(st)

    check("At least one source type has shards", len(existing_sources) > 0,
          f"Found shards for: {existing_sources}")

    if not existing_sources:
        print("\nNo shards found. Aborting.")
        sys.exit(1)

    print(f"\n  Sources with shards: {', '.join(existing_sources)}")

    for st in existing_sources:
        verify_source(st)

    # 6. Decode spot check
    print(f"\n── Decode spot check ──")
    try:
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(HF_TOKENIZER_DIR))

        import random
        random.seed(42)

        for st in existing_sources:
            train_dir = PRETOKENIZED_DIR / st / "train"
            shards = sorted(train_dir.glob("shard_*.npy"))
            shard_path = random.choice(shards)
            arr = np.load(shard_path)
            idx = random.randint(0, arr.shape[0] - 1)
            tokens = arr[idx].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            print(f"  [{st}] {shard_path.name} row {idx}")
            print(f"    tokens: {tokens[:8]}...{tokens[-4:]}")
            print(f"    text:   {text[:200]}...")
            print()
    except Exception as e:
        print(f"  \u2717 Decode check failed: {e}")
        global failed
        failed += 1

    # 7. Disk usage + stats
    total_bytes = 0
    total_tokens = 0
    total_shards = 0
    for st in existing_sources:
        train_dir = PRETOKENIZED_DIR / st / "train"
        for p in train_dir.glob("shard_*.npy"):
            arr = np.load(p)
            total_bytes += p.stat().st_size
            total_tokens += arr.shape[0] * arr.shape[1]
            total_shards += 1

    total_gb = total_bytes / (1024 ** 3)
    tokens_per_step = 4 * 64 * SEQ_LEN
    approx_steps = total_tokens // tokens_per_step

    print(f"  Disk:         {total_gb:.1f} GB")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print(f"  Total shards: {total_shards}")
    print(f"  Approx steps: {approx_steps:,}")

    # Summary
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  \u2713 All {passed} checks passed.")
    else:
        print(f"  {passed} passed, {failed} FAILED.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
