"""
BanglaGamba Tokenizer Evaluation
=================================

Compares the trained BanglaGamba tokenizer against reference tokenizers
on fertility, compression, UNK rate, and speed.

Reference tokenizers:
  - mBART-50  (facebook/mbart-large-50-many-to-many-mmt)
  - NLLB-200  (facebook/nllb-200-distilled-600M)
  - BanglaBERT (sagorsarker/bangla-bert-base)
  - GPT-2     (gpt2)

Usage:
  python scripts/util/evaluate_tokenizer.py
  python scripts/util/evaluate_tokenizer.py --sample-size 5000
  python scripts/util/evaluate_tokenizer.py --skip-references
  python scripts/util/evaluate_tokenizer.py --categories bangla english
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BANGLA_GAMBA_TOKENIZER = PROJECT_ROOT / "saved" / "tokenizer" / "hf"
BANGLA_CORPUS = PROJECT_ROOT / "saved" / "data" / "cleaned" / "bangla.jsonl"
ENGLISH_CORPUS = PROJECT_ROOT / "saved" / "data" / "cleaned" / "english.jsonl"
REPORT_DIR = PROJECT_ROOT / "saved" / "reports"

# ── Reference tokenizer model IDs ──────────────────────────────────────────

REFERENCE_MODELS = {
    "mBART-50":    "facebook/mbart-large-50-many-to-many-mmt",
    "NLLB-200":    "facebook/nllb-200-distilled-600M",
    "BanglaBERT":  "sagorsarker/bangla-bert-base",
    "GPT-2":       "gpt2",
}

# ── Curated test sentences ─────────────────────────────────────────────────

TEST_SENTENCES = {
    "bangla_formal": [
        "আমি বাংলাদেশের মানুষ। আমি বাংলায় কথা বলি।",
        "প্রধানমন্ত্রী আজ জাতীয় সংসদে ভাষণ দিয়েছেন।",
        "বাংলাদেশ একটি সুন্দর দেশ। এখানে নদী, পাহাড় এবং সমুদ্র রয়েছে।",
        "ছাত্ররা পরীক্ষার জন্য প্রস্তুতি নিচ্ছে। তারা প্রতিদিন অধ্যয়ন করে।",
        "বাংলা ভাষা আমাদের জাতীয় ভাষা। এটি বিশ্বের সবচেয়ে সুন্দর ভাষাগুলোর একটি।",
    ],
    "bangla_news": [
        "রাজধানীতে আজ সকালে একটি বিশাল অগ্নিকাণ্ড ঘটেছে।",
        "বিশ্বকাপ ফুটবলে বাংলাদেশ প্রথমবার যোগ্যতা অর্জন করেছে।",
        "চট্টগ্রাম বন্দরে নতুন টার্মিনাল উদ্বোধন করা হয়েছে।",
        "দেশে করোনা ভাইরাসের নতুন প্রকৃতি চিহ্নিত হয়েছে।",
        "সরকার নতুন শিক্ষানীতি ঘোষণা করেছে।",
    ],
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a subfield of artificial intelligence.",
        "The transformer architecture revolutionized deep learning for text.",
        "Bangladesh is a country in South Asia with a rich cultural heritage.",
        "The model was trained on a corpus of one billion words.",
    ],
    "banglish": [
        "ami tomake bhalobashi, tumi kemon acho?",
        "ajke khub bhalo din, amra park e ber korlam",
        "tumi ki kaj koro? ami ekta software engineer",
        "amar nam farhan. ami bangladeshi",
        "bhai eta ki hoise? khub bhalo lagtese",
    ],
    "code_mixed": [
        "এই product টা really ভালো, must buy করো।",
        "আমি একটি Python script লিখেছি যেটা data process করবে।",
        "তুমি কি GitHub এ code দেখেছো? সেখানে অনেক ভালো project আছে।",
        "আমাদের team এ ৫ জন developer আছে। আমরা React এ কাজ করি।",
        "বইটা খুব ভালো লেখা। তুমি কি এটা pad এ পড়েছো?",
    ],
    "python_code": [
        "def hello(): print('Hello, World!')",
        "class Model(nn.Module): def __init__(self): super().__init__()",
        "import torch; x = torch.randn(3, 3)",
        "for i in range(10): print(i ** 2)",
        "with open('data.json') as f: data = json.load(f)",
    ],
}


# ── Tokenizer loading ──────────────────────────────────────────────────────

def load_tokenizers(skip_references: bool = False) -> Dict:
    """Load BanglaGamba + reference tokenizers."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizers = {}

    # BanglaGamba
    print("Loading BanglaGamba tokenizer...")
    if BANGLA_GAMBA_TOKENIZER.exists():
        tokenizers["BanglaGamba"] = PreTrainedTokenizerFast.from_pretrained(
            str(BANGLA_GAMBA_TOKENIZER)
        )
        print(f"  [OK] BanglaGamba (vocab={tokenizers['BanglaGamba'].vocab_size})")
    else:
        print(f"  [SKIP] HF tokenizer not found at {BANGLA_GAMBA_TOKENIZER}")
        print("  Run: python -m src.tokenizer.wrapper --spm-model saved/tokenizer/banglagamba_tokenizer.model --output-dir saved/tokenizer/hf --test")

    if skip_references:
        return tokenizers

    # Reference tokenizers
    for name, model_id in REFERENCE_MODELS.items():
        try:
            print(f"Loading {name} ({model_id})...")
            tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
            print(f"  [OK] {name} (vocab={tokenizers[name].vocab_size})")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    return tokenizers


# ── Corpus sampling ────────────────────────────────────────────────────────

def sample_corpus(
    path: Path, n: int = 10_000, seed: int = 42
) -> List[str]:
    """Sample n docs from a JSONL file (reservoir sampling for large files)."""
    if not path.exists():
        print(f"  [SKIP] Corpus not found: {path}")
        return []

    random.seed(seed)
    samples = []
    total = 0

    with open(path, "r") as f:
        for line in f:
            total += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text", "").strip()
            if not text:
                continue

            if len(samples) < n:
                samples.append(text)
            else:
                j = random.randint(0, total - 1)
                if j < n:
                    samples[j] = text

    print(f"  Sampled {len(samples):,} docs from {path.name} ({total:,} total)")
    return samples


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(tokenizer, texts: List[str]) -> Dict:
    """Compute fertility, compression, UNK rate, and speed."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_unk = 0
    total_special = 0
    n = len(texts)

    t0 = time.time()
    for text in texts:
        # Tokenize without special tokens for fertility/compression
        ids = tokenizer.encode(text, add_special_tokens=False)
        unk_id = tokenizer.unk_token_id or 0

        total_tokens += len(ids)
        total_unk += sum(1 for t in ids if t == unk_id)
        total_words += max(len(text.split()), 1)
        total_chars += len(text)
    elapsed = time.time() - t0

    avg_tokens = total_tokens / max(n, 1)
    fertility = total_tokens / max(total_words, 1)
    compression = total_chars / max(total_tokens, 1)
    unk_rate = total_unk / max(total_tokens, 1) * 100
    speed = n / max(elapsed, 1e-6)

    return {
        "docs": n,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": round(avg_tokens, 1),
        "fertility": round(fertility, 3),
        "compression": round(compression, 2),
        "unk_rate_pct": round(unk_rate, 4),
        "docs_per_sec": round(speed, 1),
    }


# ── Category tests ─────────────────────────────────────────────────────────

def run_category_tests(tokenizers: Dict) -> Dict[str, Dict]:
    """Test each tokenizer on curated sentences."""
    results = {}

    for category, sentences in TEST_SENTENCES.items():
        results[category] = {}
        for tok_name, tok in tokenizers.items():
            results[category][tok_name] = compute_metrics(tok, sentences)

    return results


# ── Corpus tests ───────────────────────────────────────────────────────────

def run_corpus_tests(
    tokenizers: Dict, sample_size: int = 10_000
) -> Dict[str, Dict]:
    """Test each tokenizer on sampled corpus data."""
    print(f"\nSampling {sample_size:,} docs from each corpus...")
    bangla_samples = sample_corpus(BANGLA_CORPUS, n=sample_size)
    english_samples = sample_corpus(ENGLISH_CORPUS, n=sample_size)

    results = {}
    if bangla_samples:
        results["corpus_bangla"] = {}
        for tok_name, tok in tokenizers.items():
            results["corpus_bangla"][tok_name] = compute_metrics(tok, bangla_samples)

    if english_samples:
        results["corpus_english"] = {}
        for tok_name, tok in tokenizers.items():
            results["corpus_english"][tok_name] = compute_metrics(tok, english_samples)

    return results


# ── Output ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: Dict[str, Dict], title: str):
    """Print a formatted comparison table."""
    # Collect all tokenizer names
    all_toks = set()
    for cat_results in results.values():
        all_toks.update(cat_results.keys())
    tok_names = sorted(all_toks)

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}\n")

    # Header
    header = f"{'Category':<20}"
    for name in tok_names:
        header += f" {name:>12}"
    print(header)
    print("-" * len(header))

    # Fertility
    print(f"\n  Fertility (tokens/word, lower = better)")
    for cat, cat_results in results.items():
        row = f"  {cat:<18}"
        for name in tok_names:
            val = cat_results.get(name, {}).get("fertility", "—")
            row += f" {val:>12}" if isinstance(val, str) else f" {val:>12.3f}"
        print(row)

    # Compression
    print(f"\n  Compression (chars/token, higher = better)")
    for cat, cat_results in results.items():
        row = f"  {cat:<18}"
        for name in tok_names:
            val = cat_results.get(name, {}).get("compression", "—")
            row += f" {val:>12}" if isinstance(val, str) else f" {val:>12.2f}"
        print(row)

    # UNK rate
    print(f"\n  UNK rate % (lower = better)")
    for cat, cat_results in results.items():
        row = f"  {cat:<18}"
        for name in tok_names:
            val = cat_results.get(name, {}).get("unk_rate_pct", "—")
            row += f" {val:>12}" if isinstance(val, str) else f" {val:>12.4f}"
        print(row)

    # Speed
    print(f"\n  Speed (docs/sec)")
    for cat, cat_results in results.items():
        row = f"  {cat:<18}"
        for name in tok_names:
            val = cat_results.get(name, {}).get("docs_per_sec", "—")
            row += f" {val:>12}" if isinstance(val, str) else f" {val:>12.1f}"
        print(row)


def save_report(
    category_results: Dict, corpus_results: Dict, all_tokenizers: Dict
):
    """Save a Markdown report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "tokenizer_evaluation.md"

    lines = []
    lines.append("# BanglaGamba Tokenizer Evaluation\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("")

    # Summary
    lines.append("## Tokenizer Summary\n")
    lines.append("| Tokenizer | Vocab Size | Type |")
    lines.append("|---|---|---|")
    for name, tok in all_tokenizers.items():
        vocab = tok.vocab_size
        tok_type = "Unigram (SP)" if name == "BanglaGamba" else "Auto"
        lines.append(f"| {name} | {vocab:,} | {tok_type} |")
    lines.append("")

    # Curated tests
    lines.append("## Curated Sentence Tests\n")
    lines.append("### Fertility (tokens/word, lower = better)\n")
    all_toks = set()
    for cat_results in category_results.values():
        all_toks.update(cat_results.keys())
    tok_names = sorted(all_toks)

    header = "| Category |"
    sep = "|---|"
    for name in tok_names:
        header += f" {name} |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for cat, cat_results in category_results.items():
        row = f"| {cat} |"
        for name in tok_names:
            val = cat_results.get(name, {}).get("fertility", "—")
            row += f" {val:.3f} |" if isinstance(val, (int, float)) else f" {val} |"
        lines.append(row)
    lines.append("")

    lines.append("### Compression (chars/token, higher = better)\n")
    lines.append(header)
    lines.append(sep)
    for cat, cat_results in category_results.items():
        row = f"| {cat} |"
        for name in tok_names:
            val = cat_results.get(name, {}).get("compression", "—")
            row += f" {val:.2f} |" if isinstance(val, (int, float)) else f" {val} |"
        lines.append(row)
    lines.append("")

    lines.append("### UNK Rate % (lower = better)\n")
    lines.append(header)
    lines.append(sep)
    for cat, cat_results in category_results.items():
        row = f"| {cat} |"
        for name in tok_names:
            val = cat_results.get(name, {}).get("unk_rate_pct", "—")
            row += f" {val:.4f} |" if isinstance(val, (int, float)) else f" {val} |"
        lines.append(row)
    lines.append("")

    # Corpus tests
    if corpus_results:
        lines.append("## Corpus Tests (sampled from cleaned data)\n")
        lines.append("### Fertility (tokens/word)\n")
        all_toks = set()
        for cat_results in corpus_results.values():
            all_toks.update(cat_results.keys())
        tok_names = sorted(all_toks)

        header = "| Corpus |"
        sep = "|---|"
        for name in tok_names:
            header += f" {name} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        for cat, cat_results in corpus_results.items():
            row = f"| {cat} |"
            for name in tok_names:
                val = cat_results.get(name, {}).get("fertility", "—")
                row += f" {val:.3f} |" if isinstance(val, (int, float)) else f" {val} |"
            lines.append(row)
        lines.append("")

    # Detailed metrics
    lines.append("## Detailed Metrics\n")
    for category_results_set in [category_results, corpus_results]:
        for cat, cat_results in category_results_set.items():
            lines.append(f"### {cat}\n")
            lines.append("| Metric |" + "".join(f" {n} |" for n in sorted(cat_results.keys())))
            lines.append("|---|" + "".join("---|" for _ in cat_results))
            for metric in ["fertility", "compression", "unk_rate_pct", "avg_tokens_per_doc", "docs_per_sec"]:
                row = f"| {metric} |"
                for name in sorted(cat_results.keys()):
                    val = cat_results[name].get(metric, "—")
                    row += f" {val} |"
                lines.append(row)
            lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"\n[OK] Report saved to: {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BanglaGamba tokenizer against references.")
    parser.add_argument("--sample-size", type=int, default=10_000,
                        help="Number of corpus docs to sample per language.")
    parser.add_argument("--skip-references", action="store_true",
                        help="Skip downloading reference tokenizers (test BanglaGamba only).")
    parser.add_argument("--categories", nargs="*",
                        help="Run only specific category tests (e.g., bangla_formal english).")
    parser.add_argument("--no-corpus", action="store_true",
                        help="Skip corpus-based tests.")
    args = parser.parse_args()

    print("=" * 60)
    print("  BanglaGamba Tokenizer Evaluation")
    print("=" * 60)

    # Load tokenizers
    tokenizers = load_tokenizers(skip_references=args.skip_references)
    if not tokenizers:
        print("ERROR: No tokenizers loaded. Run wrapper first.")
        sys.exit(1)

    # Category tests
    category_results = {}
    if args.categories:
        for cat in args.categories:
            if cat in TEST_SENTENCES:
                category_results[cat] = {}
                for tok_name, tok in tokenizers.items():
                    category_results[cat][tok_name] = compute_metrics(tok, TEST_SENTENCES[cat])
    else:
        category_results = run_category_tests(tokenizers)

    # Corpus tests
    corpus_results = {}
    if not args.no_corpus:
        corpus_results = run_corpus_tests(tokenizers, sample_size=args.sample_size)

    # Print results
    print_comparison_table(category_results, "Curated Sentence Tests")
    if corpus_results:
        print_comparison_table(corpus_results, "Corpus Tests")

    # Save report
    save_report(category_results, corpus_results, tokenizers)

    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
