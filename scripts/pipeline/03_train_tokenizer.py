"""
BanglaGamba — Train Tokenizer
================================
Sample from cleaned corpus, train 48K SentencePiece BPE tokenizer,
wrap as HF tokenizer. Delete tokenizer training corpus after.

Usage:
  python scripts/pipeline/03_train_tokenizer.py
  python scripts/pipeline/03_train_tokenizer.py --target-gb 3.0
  python scripts/pipeline/03_train_tokenizer.py --vocab-size 32000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm


CLEANED_PATH = Path("saved/data/cleaned/corpus_cleaned.jsonl")
TOKENIZER_CORPUS = Path("saved/data/tokenizer/tokenizer_corpus.txt")
SP_DIR = Path("saved/tokenizer/sp")
HF_DIR = Path("saved/tokenizer/hf")

# Target sampling in MB per source_type (sum ~5 GB)
SOURCE_SAMPLE_MB = {
    "web_bangla":         2500,
    "romanized_bangla":    800,
    "encyclopedic":        500,
    "parallel_bn_en":      600,
    "code_mixed_informal": 300,
}


def sample_tokenizer_corpus(target_gb: float = 5.0):
    """Read cleaned corpus, sample proportionally by source_type, write txt."""
    TOKENIZER_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    if not CLEANED_PATH.exists():
        print(f"[tokenizer] ERROR: {CLEANED_PATH} not found. Run 02_clean.py first.")
        sys.exit(1)

    total_bytes = CLEANED_PATH.stat().st_size
    target_bytes = target_gb * 1024 ** 3
    scale = target_bytes / max(total_bytes, 1)

    targets_bytes = {k: int(v * 1024 ** 2 * scale) for k, v in SOURCE_SAMPLE_MB.items()}
    collected_bytes = {k: 0 for k in SOURCE_SAMPLE_MB}

    print(f"[tokenizer] Sampling ~{target_gb:.1f} GB from {CLEANED_PATH}")
    print(f"[tokenizer] Targets:")
    for k, v in targets_bytes.items():
        print(f"            {k:25s}  {v / 1024**2:,.0f} MB")

    with open(CLEANED_PATH, "r") as fin, open(TOKENIZER_CORPUS, "w") as fout:
        line_count = 0
        written = 0
        with tqdm(total=total_bytes, desc="Sampling", unit="B", unit_scale=True) as bar:
            for line in fin:
                bar.update(len(line.encode("utf-8")))
                line_count += 1

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stype = doc.get("source_type", "")
                text = doc.get("text", "")

                if stype not in targets_bytes:
                    continue
                if collected_bytes[stype] >= targets_bytes[stype]:
                    continue

                # Collapse internal newlines to space
                text = " ".join(text.split())
                text_bytes = len(text.encode("utf-8")) + 1  # +1 for newline

                fout.write(text + "\n")
                collected_bytes[stype] += text_bytes
                written += 1

    total_sampled = sum(collected_bytes.values())
    print(f"\n[tokenizer] Sampled {written:,} docs, {total_sampled / 1024**2:,.1f} MB")
    for k, v in collected_bytes.items():
        print(f"            {k:25s}  {v / 1024**2:,.1f} MB")


def train_sentencepiece(vocab_size: int = 48000):
    """Train SentencePiece BPE model."""
    SP_DIR.mkdir(parents=True, exist_ok=True)

    if not TOKENIZER_CORPUS.exists():
        print(f"[tokenizer] ERROR: {TOKENIZER_CORPUS} not found.")
        sys.exit(1)

    print(f"[tokenizer] Training SentencePiece BPE (vocab={vocab_size})...")

    spm.SentencePieceTrainer.train(
        input=str(TOKENIZER_CORPUS),
        model_prefix=str(SP_DIR / "banglagamba"),
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=[
            "<|lang_bn|>",
            "<|lang_bnls|>",
            "<|lang_en|>",
            "<|lang_mix|>",
            "|SEP|",
        ],
        byte_fallback=True,
        split_digits=True,
        num_threads=8,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )

    print(f"[tokenizer] SP model saved: {SP_DIR / 'banglagamba.model'}")


def wrap_hf_tokenizer():
    """Convert SP model to HF PreTrainedTokenizerFast."""
    # Add project root to path for imports
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.tokenizer.wrapper import create_hf_tokenizer

    sp_model = SP_DIR / "banglagamba.model"
    if not sp_model.exists():
        print(f"[tokenizer] ERROR: SP model not found: {sp_model}")
        sys.exit(1)

    create_hf_tokenizer(
        spm_model_path=str(sp_model),
        output_dir=str(HF_DIR),
    )


def print_sanity_encodes():
    """Print visual sanity check of encodes."""
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(HF_DIR))

    test_sentences = [
        "আমি বাংলায় গান গাই",
        "ami ektu tired feel korchi",
        "আমার laptop টা crash করলো",
    ]

    print("\nSanity encodes:")
    for sent in test_sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        ids_str = ", ".join(str(x) for x in ids[:12])
        if len(ids) > 12:
            ids_str += ", ..."
        print(f'  "{sent}"')
        print(f"    → [{ids_str}]")
        print(f'    → decoded: "{decoded}"')
        print()

    # Special token IDs
    sep_id = tokenizer.convert_tokens_to_ids("|SEP|")
    lang_bn_id = tokenizer.convert_tokens_to_ids("<|lang_bn|>")
    print(f'  |SEP| token id:            → {sep_id}')
    print(f'  <|lang_bn|> token id:      → {lang_bn_id}')


def main():
    parser = argparse.ArgumentParser(description="Train 48K BPE tokenizer.")
    parser.add_argument("--target-gb", type=float, default=5.0,
                        help="Target size of tokenizer training corpus in GB.")
    parser.add_argument("--vocab-size", type=int, default=48000,
                        help="SentencePiece vocab size.")
    args = parser.parse_args()

    sample_tokenizer_corpus(target_gb=args.target_gb)
    train_sentencepiece(vocab_size=args.vocab_size)
    wrap_hf_tokenizer()

    # Delete tokenizer training corpus
    if TOKENIZER_CORPUS.exists():
        size_gb = TOKENIZER_CORPUS.stat().st_size / (1024 ** 3)
        TOKENIZER_CORPUS.unlink()
        print(f"[tokenizer] Deleted training corpus — freed {size_gb:.1f} GB")

    print("\n" + "=" * 50)
    print("=== TOKENIZER TRAINED ===")
    print(f"Vocab size:  {args.vocab_size:,}")
    print(f"SP model:    {SP_DIR / 'banglagamba.model'}")
    print(f"HF path:     {HF_DIR}")
    print("=" * 50)

    print_sanity_encodes()


if __name__ == "__main__":
    main()
