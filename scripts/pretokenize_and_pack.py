"""
Pretokenize & Pack — per source type.

Tokenizes every document, packs into 2048-token sequences, writes .npy shards.
Output directories:
  saved/data/pretokenized/bangla/train/
  saved/data/pretokenized/english/train/
  saved/data/pretokenized/nmt/train/
  saved/data/pretokenized/sangraha/train/

No language token injection — tokens are already in text from downloaders.
No eval split — everything goes to train.

Usage:
  python scripts/pretokenize_and_pack.py
  python scripts/pretokenize_and_pack.py --delete-cleaned
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


CLEANED_DIR = Path("saved/data/cleaned")
HF_TOKENIZER_DIR = Path("saved/tokenizer/hf")
PRETOKENIZED_DIR = Path("saved/data/pretokenized")

SEQ_LEN = 2048
BATCH_TOKENS = SEQ_LEN * 100_000  # 204.8M tokens per shard

# Source type → input files + output directory
SOURCE_CONFIGS = {
    "bangla": {
        "inputs": [
            CLEANED_DIR / "bangla.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "bangla" / "train",
    },
    "english": {
        "inputs": [
            CLEANED_DIR / "english.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "english" / "train",
    },
    "nmt": {
        "inputs": [
            CLEANED_DIR / "nmt.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "nmt" / "train",
    },
    "opus_nmt": {
        "inputs": [
            CLEANED_DIR / "opus_nmt.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "opus_nmt" / "train",
    },
    "sangraha": {
        "inputs": [
            CLEANED_DIR / "sangraha.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "sangraha" / "train",
    },
}


def _setup_imports():
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_tokenizer():
    _setup_imports()
    from transformers import PreTrainedTokenizerFast

    if not HF_TOKENIZER_DIR.exists():
        print(f"[pretokenize] ERROR: HF tokenizer not found at {HF_TOKENIZER_DIR}")
        sys.exit(1)

    return PreTrainedTokenizerFast.from_pretrained(str(HF_TOKENIZER_DIR))


def save_shard(token_ids: list[int], shard_idx: int, output_dir: Path) -> int:
    usable = len(token_ids) - (len(token_ids) % SEQ_LEN)
    if usable == 0:
        return 0
    arr = np.array(token_ids[:usable], dtype=np.uint16).reshape(-1, SEQ_LEN)
    shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
    tmp_path = shard_path.with_suffix(".tmp.npy")
    np.save(tmp_path, arr)
    tmp_path.replace(shard_path)
    return arr.shape[0]


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def _resolve_inputs(config: dict) -> list[Path]:
    """Return the configured input files."""
    return [p for p in config["inputs"] if p.exists()]


def pretokenize_source(
    source_type: str,
    config: dict,
    tokenizer,
) -> tuple[int, int]:
    """Pretokenize one source type. Returns (tokens_written, docs_processed)."""
    output_dir = config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(config)
    if not inputs:
        print(f"[pretokenize] ERROR: input file(s) missing for {source_type}: "
              f"{[str(p) for p in config['inputs']]}")
        sys.exit(1)

    # Count total lines
    total_lines = sum(_count_lines(p) for p in inputs)

    eos_id = tokenizer.eos_token_id
    buffer = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    print(f"[pretokenize] {source_type}: {len(inputs)} input file(s), {total_lines:,} lines")

    with tqdm(total=total_lines, desc=f"  {source_type}", unit="docs", unit_scale=True) as bar:
        for input_path in inputs:
            with open(input_path, "r") as f:
                for line in f:
                    bar.update(1)

                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = doc.get("text", "").strip()
                    if not text:
                        continue

                    # Tokenize — text already has special tokens from downloaders
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    tokens = tokens + [eos_id]

                    buffer.extend(tokens)
                    total_tokens += len(tokens)
                    total_docs += 1

                    # Flush when buffer is large enough
                    while len(buffer) >= BATCH_TOKENS:
                        chunk = buffer[:BATCH_TOKENS]
                        buffer = buffer[BATCH_TOKENS:]
                        save_shard(chunk, shard_idx, output_dir)
                        shard_idx += 1

    # Save remaining buffer
    if buffer:
        remainder = len(buffer) % SEQ_LEN
        if remainder:
            print(f"  [pretokenize] Truncating final buffer: discarding {remainder} tokens")
        save_shard(buffer, shard_idx, output_dir)
        shard_idx += 1

    return total_tokens, total_docs


def main():
    parser = argparse.ArgumentParser(description="Pretokenize and pack into .npy shards.")
    parser.add_argument("--delete-cleaned", action="store_true",
                        help="Delete cleaned/ directory after pretokenization.")
    parser.add_argument("--source", choices=["bangla", "english", "nmt", "opus_nmt", "sangraha", "all"], default="all",
                        help="Which source type to pretokenize (default: all).")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    print(f"[pretokenize] Tokenizer loaded (vocab={tokenizer.vocab_size})")

    sources = list(SOURCE_CONFIGS.keys()) if args.source == "all" else [args.source]

    grand_tokens = 0
    grand_docs = 0

    for source_type in sources:
        config = SOURCE_CONFIGS[source_type]
        tokens, docs = pretokenize_source(source_type, config, tokenizer)
        grand_tokens += tokens
        grand_docs += docs

    # Calculate stats
    total_shards = 0
    for source_type in sources:
        output_dir = SOURCE_CONFIGS[source_type]["output"]
        shards = list(output_dir.glob("shard_*.npy"))
        total_shards += len(shards)

    tokens_per_step = 4 * 64 * SEQ_LEN
    approx_steps = grand_tokens // tokens_per_step

    print(f"\n{'=' * 50}")
    print(f"=== PRETOKENIZATION COMPLETE ===")
    print(f"  Documents:       {grand_docs:,}")
    print(f"  Tokens:          {grand_tokens:,}  ({grand_tokens / 1e9:.2f}B)")
    print(f"  Total shards:    {total_shards}")
    for source_type in sources:
        output_dir = SOURCE_CONFIGS[source_type]["output"]
        n = len(list(output_dir.glob("shard_*.npy")))
        print(f"    {source_type:>10}: {n:>4} shards  →  {output_dir}")
    print(f"  Approx steps:    {approx_steps:,}")
    print(f"    (batch=4 × accum=64 × seq=2048 = {tokens_per_step:,} tokens/step)")
    print(f"\n  ACTION: set max_steps: {approx_steps} in configs/default_training.yaml")
    print(f"{'=' * 50}")

    # Delete cleaned (opt-in)
    if args.delete_cleaned and CLEANED_DIR.exists():
        shutil.rmtree(CLEANED_DIR)
        print(f"[pretokenize] Deleted {CLEANED_DIR}")


if __name__ == "__main__":
    main()
