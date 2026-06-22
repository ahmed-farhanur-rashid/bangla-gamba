"""
BanglaGamba — Pretokenize & Pack
===================================
Tokenize every document, pack into 2048-token sequences, write .npy shards.

Usage:
  python scripts/pipeline/04_pretokenize.py
  python scripts/pipeline/04_pretokenize.py --delete-cleaned
  python scripts/pipeline/04_pretokenize.py --max-tokens 1_000_000_000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


CLEANED_PATH = Path("saved/data/cleaned/corpus_deduped.jsonl")
HF_TOKENIZER_DIR = Path("saved/tokenizer/hf")
PRETOKENIZED_DIR = Path("saved/data/pretokenized")
TRAIN_DIR = PRETOKENIZED_DIR / "train"
EVAL_DIR = PRETOKENIZED_DIR / "eval"

SEQ_LEN = 2048
BATCH_TOKENS = SEQ_LEN * 100_000  # 204.8M tokens per shard


def _setup_imports():
    """Ensure project root is on sys.path for src imports."""
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_tokenizer():
    """Load the HF tokenizer."""
    _setup_imports()
    from transformers import PreTrainedTokenizerFast

    if not HF_TOKENIZER_DIR.exists():
        print(f"[pretokenize] ERROR: HF tokenizer not found at {HF_TOKENIZER_DIR}")
        sys.exit(1)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(HF_TOKENIZER_DIR))
    return tokenizer


def save_shard(token_ids: list[int], shard_idx: int, output_dir: Path):
    """Truncate to nearest multiple of SEQ_LEN, reshape to (N, 2048), save."""
    usable = len(token_ids) - (len(token_ids) % SEQ_LEN)
    if usable == 0:
        return 0
    arr = np.array(token_ids[:usable], dtype=np.uint16).reshape(-1, SEQ_LEN)
    shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
    np.save(shard_path, arr)
    return arr.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Pretokenize and pack into .npy shards.")
    parser.add_argument("--delete-cleaned", action="store_true",
                        help="Delete cleaned/ directory after pretokenization.")
    parser.add_argument("--max-tokens", type=int, default=10_000_000_000,
                        help="Stop after this many tokens (default: 10B).")
    args = parser.parse_args()

    if not CLEANED_PATH.exists():
        print(f"[pretokenize] ERROR: {CLEANED_PATH} not found. Run 02_clean.py first.")
        sys.exit(1)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    print(f"[pretokenize] Tokenizer loaded (vocab={vocab_size}, eos_id={eos_id})")
    print(f"[pretokenize] Reading {CLEANED_PATH}...")

    # Count total lines for progress
    total_lines = 0
    with open(CLEANED_PATH, "rb") as f:
        for _ in f:
            total_lines += 1

    # Tokenize and pack
    from src.tokenizer.special_tokens import get_lang_token

    buffer = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    with open(CLEANED_PATH, "r") as f, tqdm(total=total_lines, desc="Pretokenizing", unit="docs", unit_scale=True) as bar:
        for line in f:
            bar.update(1)
            if total_tokens >= args.max_tokens:
                break

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "")
            language_region = doc.get("language_region", "BD_WB_mix")

            # Get language token via canonical mapping
            lang_token_str = get_lang_token(language_region)
            lang_token_id = tokenizer.convert_tokens_to_ids(lang_token_str)

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens = [lang_token_id] + tokens + [eos_id]

            buffer.extend(tokens)
            total_tokens += len(tokens)
            total_docs += 1

            # Flush when buffer is large enough
            while len(buffer) >= BATCH_TOKENS:
                chunk = buffer[:BATCH_TOKENS]
                buffer = buffer[BATCH_TOKENS:]

                if shard_idx < 2:
                    n_rows = save_shard(chunk, shard_idx, EVAL_DIR)
                else:
                    n_rows = save_shard(chunk, shard_idx, TRAIN_DIR)
                shard_idx += 1

        # Save remaining buffer (truncated to nearest SEQ_LEN by save_shard)
        if buffer and total_tokens <= args.max_tokens:
            remainder = len(buffer) % SEQ_LEN
            if remainder:
                print(f"[pretokenize] Truncating final buffer: discarding {remainder} tokens")
            if shard_idx < 2:
                save_shard(buffer, shard_idx, EVAL_DIR)
            else:
                save_shard(buffer, shard_idx, TRAIN_DIR)
            shard_idx += 1

    # Verify at least 1 shard in train
    train_shards = list(TRAIN_DIR.glob("shard_*.npy"))
    eval_shards = list(EVAL_DIR.glob("shard_*.npy"))

    if not train_shards:
        print("[pretokenize] WARNING: No train shards created!")

    # Delete cleaned (opt-in only)
    if args.delete_cleaned and CLEANED_PATH.exists():
        cleaned_size = CLEANED_PATH.stat().st_size / (1024 ** 3)
        shutil.rmtree(CLEANED_PATH.parent)
        print(f"[pretokenize] Deleted {CLEANED_PATH.parent} — freed {cleaned_size:.1f} GB")

    # Calculate disk usage
    total_disk = sum(f.stat().st_size for f in PRETOKENIZED_DIR.rglob("*.npy")) / (1024 ** 3)

    # Training steps estimate
    tokens_per_step = 4 * 64 * SEQ_LEN  # batch_size=4, accum=64
    approx_steps = total_tokens // tokens_per_step

    print("\n" + "=" * 50)
    print("=== PRETOKENIZATION COMPLETE ===")
    print(f"Documents:        {total_docs:,}")
    print(f"Tokens:           {total_tokens:,}  ({total_tokens / 1e9:.2f}B)")
    print(f"Train shards:     {len(train_shards):>4}  →  {TRAIN_DIR}")
    print(f"Eval shards:      {len(eval_shards):>4}  →  {EVAL_DIR}")
    print(f"Disk:             ~{total_disk:.1f} GB")
    print()
    print(f"Approx training steps: {approx_steps:,}")
    print(f"  (tokens_per_step = batch_size=4 × accum=64 × seq_len=2048 = {tokens_per_step:,})")
    print()
    print(f"ACTION REQUIRED: set  max_steps: {approx_steps}  in configs/default_training.yaml")
    print("=" * 50)


if __name__ == "__main__":
    main()
