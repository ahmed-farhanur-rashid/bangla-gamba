"""
Pretokenize & Pack — per source type.

Tokenizes every document, packs into 2048-token sequences, writes .npy shards.
Output directories:
  saved/data/pretokenized/bangla/train/
  saved/data/pretokenized/english/train/
  saved/data/pretokenized/nmt/train/

No language token injection — tokens are already in text from downloaders.
No eval split — everything goes to train.

Usage:
  python scripts/pipeline/04_pretokenize.py
  python scripts/pipeline/04_pretokenize.py --max-tokens 5_000_000_000
  python scripts/pipeline/04_pretokenize.py --delete-cleaned
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
RAW_DIR = Path("saved/data/raw")
HF_TOKENIZER_DIR = Path("saved/tokenizer/hf")
PRETOKENIZED_DIR = Path("saved/data/pretokenized")

SEQ_LEN = 2048
BATCH_TOKENS = SEQ_LEN * 100_000  # 204.8M tokens per shard

# Source type → input files + output directory
SOURCE_CONFIGS = {
    "bangla": {
        "inputs": [
            CLEANED_DIR / "bangla.jsonl",  # deduped TituLLM + Wiki
        ],
        "fallback_inputs": [
            RAW_DIR / "titullm.jsonl",     # if dedup not run yet
            RAW_DIR / "wiki_bangla.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "bangla" / "train",
    },
    "english": {
        "inputs": [
            RAW_DIR / "fineweb_edu.jsonl",
        ],
        "fallback_inputs": [],
        "output": PRETOKENIZED_DIR / "english" / "train",
    },
    "nmt": {
        "inputs": [
            CLEANED_DIR / "nmt.jsonl",     # deduped NLLB + BanglaNMT
        ],
        "fallback_inputs": [
            RAW_DIR / "nllb.jsonl",        # if dedup not run yet
            RAW_DIR / "banglanmt.jsonl",
        ],
        "output": PRETOKENIZED_DIR / "nmt" / "train",
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
    np.save(shard_path, arr)
    return arr.shape[0]


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def _resolve_inputs(config: dict) -> list[Path]:
    """Return existing input files, preferring cleaned over raw."""
    inputs = [p for p in config["inputs"] if p.exists()]
    if not inputs:
        inputs = [p for p in config.get("fallback_inputs", []) if p.exists()]
    return inputs


def pretokenize_source(
    source_type: str,
    config: dict,
    tokenizer,
    max_tokens: int,
) -> tuple[int, int]:
    """Pretokenize one source type. Returns (tokens_written, docs_processed)."""
    output_dir = config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(config)
    if not inputs:
        print(f"[pretokenize] WARNING: No input files for {source_type}, skipping")
        return 0, 0

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
                    if total_tokens >= max_tokens:
                        break

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
                        rows = save_shard(chunk, shard_idx, output_dir)
                        shard_idx += 1

            if total_tokens >= max_tokens:
                break

    # Save remaining buffer
    if buffer:
        remainder = len(buffer) % SEQ_LEN
        if remainder:
            print(f"  [pretokenize] Truncating final buffer: discarding {remainder} tokens")
        rows = save_shard(buffer, shard_idx, output_dir)
        shard_idx += 1

    return total_tokens, total_docs


def main():
    parser = argparse.ArgumentParser(description="Pretokenize and pack into .npy shards.")
    parser.add_argument("--max-tokens", type=int, default=10_000_000_000,
                        help="Stop after this many tokens (default: 10B).")
    parser.add_argument("--delete-cleaned", action="store_true",
                        help="Delete cleaned/ directory after pretokenization.")
    parser.add_argument("--source", choices=["bangla", "english", "nmt", "all"], default="all",
                        help="Which source type to pretokenize (default: all).")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    print(f"[pretokenize] Tokenizer loaded (vocab={tokenizer.vocab_size})")

    sources = list(SOURCE_CONFIGS.keys()) if args.source == "all" else [args.source]

    grand_tokens = 0
    grand_docs = 0

    for source_type in sources:
        config = SOURCE_CONFIGS[source_type]
        tokens, docs = pretokenize_source(source_type, config, tokenizer, args.max_tokens)
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
