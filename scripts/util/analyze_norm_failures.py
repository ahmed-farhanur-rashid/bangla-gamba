#!/usr/bin/env python3
"""
Analyze normalization failure logs and generate a YAML report.

Reads the JSONL failure log produced by 01d_bn_normalize.py with --none-policy drop_and_collect.
Outputs a structured YAML report to saved/reports/.

Usage:
    python -m scripts.util.analyze_norm_failures

    python -m scripts.util.analyze_norm_failures
        --input saved/logs/bangla_deduped_norm_failures.jsonl
        
    python3 -m scripts.util.analyze_norm_failures
        --input saved/logs/bangla_deduped_norm_failures.jsonl
        --output saved/reports/bangla_deduped_norm_failures.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

LOG_DIR = Path("saved/logs")
REPORT_DIR = Path("saved/reports")

# Bengali diacritics / signs that are valid Unicode but invalid as standalone tokens
BENGALI_VOWEL_SIGNS = {
    "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ", "ো", "ৌ", "া",  # dependent vowel signs
}
BENGALI_NASAL_MARKS = {"ঁ", "ং", "ঃ"}
BENGALI_SYMBOLS = {"৳", "৴", "৵", "৶", "৷", "৹", "ৢ", "ৣ"}
BENGALI_HALANT = "্"


def parse_args():
    p = argparse.ArgumentParser(description="Analyze normalization failure logs.")
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Failure log JSONL (default: latest in saved/logs/)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path (default: saved/reports/norm_failures.yaml)",
    )
    return p.parse_args()


def classify_token(token: str) -> str:
    """Classify a failed token into a category."""
    if len(token) == 1:
        ch = token
        if ch in BENGALI_VOWEL_SIGNS:
            return "standalone_vowel_sign"
        if ch in BENGALI_NASAL_MARKS:
            return "standalone_nasalization"
        if ch in BENGALI_SYMBOLS:
            return "symbol"
        if ch == BENGALI_HALANT:
            return "standalone_halant"
        # other single char
        from unicodedata import category
        cat = category(ch)
        if cat.startswith("M"):  # Mark
            return "standalone_mark"
        if cat.startswith("S"):  # Symbol
            return "symbol"
        return "other_single_char"

    # Multi-char: check if repeated single char
    unique_chars = set(token)
    if len(unique_chars) == 1:
        return "repeated_char"

    # Multi-char: check if all vowel signs
    if all(ch in BENGALI_VOWEL_SIGNS for ch in token):
        return "stacked_vowel_signs"
    if all(ch in BENGALI_NASAL_MARKS for ch in token):
        return "stacked_nasal_marks"

    # Mixed diacritics + other
    if any(ch in BENGALI_VOWEL_SIGNS or ch in BENGALI_NASAL_MARKS for ch in token):
        return "mixed_diacritics"

    return "other_multi_char"


def analyze(input_path: Path) -> dict:
    tokens = Counter()
    sources = Counter()
    token_sources = Counter()  # (token, source) pairs
    categories = Counter()
    token_details = {}

    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            t = d["token"]
            s = d["source"]
            tokens[t] += 1
            sources[s] += 1
            token_sources[(t, s)] += 1
            cat = classify_token(t)
            categories[cat] += 1
            if t not in token_details:
                token_details[t] = {
                    "unicode": f"U+{ord(t[0]):04X}",
                    "name": _char_name(t[0]),
                    "length": len(t),
                    "classification": cat,
                }

    total_failures = sum(tokens.values())

    # Build top tokens list
    top_tokens = []
    for token, count in tokens.most_common():
        detail = token_details[token]
        sources_for_token = {}
        for s in sources:
            c = token_sources.get((token, s), 0)
            if c > 0:
                sources_for_token[s] = c
        top_tokens.append({
            "token": token,
            "count": count,
            "unicode": detail["unicode"],
            "name": detail["name"],
            "length": detail["length"],
            "classification": detail["classification"],
            "sources": sources_for_token,
        })

    # Component analysis for multi-char failures
    component_counts = Counter()
    for token, count in tokens.items():
        if len(token) > 1:
            for ch in set(token):
                component_counts[ch] += count

    component_analysis = []
    for ch, count in component_counts.most_common():
        component_analysis.append({
            "token": ch,
            "unicode": f"U+{ord(ch):04X}",
            "name": _char_name(ch),
            "count": count,
        })

    report = {
        "summary": {
            "input_file": str(input_path),
            "total_failures": total_failures,
            "unique_tokens": len(tokens),
        },
        "sources": dict(sources.most_common()),
        "classification": dict(categories.most_common()),
        "top_tokens": top_tokens,
        "component_analysis": component_analysis,
    }

    return report


def _char_name(ch: str) -> str:
    """Get Unicode character name, fallback to codepoint."""
    import unicodedata
    try:
        return unicodedata.name(ch, "")
    except ValueError:
        return ""


def main():
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        logs = sorted(LOG_DIR.glob("*_norm_failures.jsonl"))
        if not logs:
            print("No failure logs found in saved/logs/")
            return
        input_path = logs[-1]
        print(f"Using: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPORT_DIR / f"{input_path.stem}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = analyze(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Report written to {output_path}")
    print(f"  Total failures: {report['summary']['total_failures']:,}")
    print(f"  Unique tokens: {report['summary']['unique_tokens']}")
    print(f"  Sources: {report['sources']}")
    print(f"  Classifications: {report['classification']}")


if __name__ == "__main__":
    main()
