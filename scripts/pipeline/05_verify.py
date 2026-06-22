"""
BanglaGamba — Verify Shards
================================
Fast read-only sanity check before starting training.

Usage:
  python scripts/pipeline/05_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PRETOKENIZED_DIR = Path("saved/data/pretokenized")
TRAIN_DIR = PRETOKENIZED_DIR / "train"
EVAL_DIR = PRETOKENIZED_DIR / "eval"
HF_TOKENIZER_DIR = Path("saved/tokenizer/hf")

SEQ_LEN = 2048
VOCAB_SIZE = 48000

passed = 0
failed = 0


def check(name: str, ok: bool, msg: str):
    global passed, failed
    if ok:
        print(f"  \u2713 {name}")
        passed += 1
    else:
        print(f"  \u2717 {name}: {msg}")
        failed += 1


def main():
    print("=" * 60)
    print("  BanglaGamba — Shard Verification")
    print("=" * 60 + "\n")

    # 1. Shard count
    train_shards = sorted(TRAIN_DIR.glob("shard_*.npy")) if TRAIN_DIR.exists() else []
    eval_shards = sorted(EVAL_DIR.glob("shard_*.npy")) if EVAL_DIR.exists() else []
    all_shards = train_shards + eval_shards

    check("Train shards exist", len(train_shards) > 0,
          f"Found {len(train_shards)} shards in {TRAIN_DIR}")
    check("Eval shards exist", len(eval_shards) > 0,
          f"Found {len(eval_shards)} shards in {EVAL_DIR}")

    if not all_shards:
        print("\nNo shards found. Aborting verification.")
        sys.exit(1)

    print(f"  Total shards: {len(train_shards)} train + {len(eval_shards)} eval\n")

    # 2. Shape check — first and last train shard
    for shard_path in [train_shards[0], train_shards[-1]]:
        arr = np.load(shard_path)
        check(f"Shape {shard_path.name} == (N, {SEQ_LEN})",
              arr.ndim == 2 and arr.shape[1] == SEQ_LEN,
              f"Got shape {arr.shape}")
        check(f"dtype {shard_path.name} == uint16",
              arr.dtype == np.uint16,
              f"Got dtype {arr.dtype}")

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
          f"{zero_rows}/{total_rows} rows are all zeros ({zero_pct:.1f}%)")

    # 5. Decode spot check — 3 random sequences from 3 random shards
    print()
    try:
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(HF_TOKENIZER_DIR))

        import random
        random.seed(42)
        sample_shards = random.sample(all_shards, min(3, len(all_shards)))

        print("  Decode spot checks:")
        for shard_path in sample_shards:
            arr = np.load(shard_path)
            idx = random.randint(0, arr.shape[0] - 1)
            tokens = arr[idx].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            print(f"    [{shard_path.name} row {idx}]")
            print(f"    tokens: {tokens[:8]}...{tokens[-4:]}")
            print(f"    text:   {text[:200]}...")
            print()
    except Exception as e:
        print(f"  \u2717 Decode spot check failed: {e}")
        global failed
        failed += 1

    # 6. Disk usage
    total_bytes = sum(f.stat().st_size for f in PRETOKENIZED_DIR.rglob("*.npy"))
    total_gb = total_bytes / (1024 ** 3)
    print(f"  Disk usage: {total_gb:.1f} GB")

    # 7. Training steps
    total_tokens = 0
    for shard_path in all_shards:
        arr = np.load(shard_path)
        total_tokens += arr.shape[0] * arr.shape[1]

    tokens_per_step = 4 * 64 * SEQ_LEN  # batch_size=4, accum=64
    approx_steps = total_tokens // tokens_per_step

    print(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print(f"  Approx training steps: {approx_steps:,}")

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
