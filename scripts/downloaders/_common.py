"""Shared utilities for download scripts (v3 schema)."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

RAW_DIR = Path("saved/data/raw")
CLEANED_DIR = Path("saved/data/cleaned")

# ── Token prefixes (source of truth: src/tokenizer/special_tokens.py) ────────
LANG_BN    = "<|lang_bn|>"
LANG_EN    = "<|lang_en|>"
TASK_BN_EN = "<|task_translate_bn_en|>"
TASK_EN_BN = "<|task_translate_en_bn|>"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def write_doc(f, text: str, source: str, source_type: str,
              language_region: str, word_count: int):
    f.write(json.dumps({
        "text": text,
        "source": source,
        "source_type": source_type,
        "language_region": language_region,
        "word_count": word_count,
    }, ensure_ascii=False) + "\n")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    """For NMT pairs — collapses whitespace, single-line output."""
    return " ".join(nfc(text).split())


def normalize_doc(text: str) -> str:
    """For monolingual docs — preserves paragraph breaks."""
    text = nfc(text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def wc(text: str) -> int:
    return len(text.split())


def has_min_words(text: str, min_words: int = 20) -> bool:
    return wc(text) >= min_words


# ── NMT helpers ──────────────────────────────────────────────────────────────

def length_ok(bn: str, en: str) -> bool:
    b, e = wc(bn), wc(en)
    return 3 <= b <= 150 and 3 <= e <= 150


def ratio_ok(bn: str, en: str) -> bool:
    b = wc(bn)
    if b == 0:
        return False
    return 0.4 <= wc(en) / b <= 2.5


seen_bn: set[bytes] = set()


def is_dup_nmt(bn: str) -> bool:
    h = hashlib.md5(bn.encode()).digest()
    if h in seen_bn:
        return True
    seen_bn.add(h)
    return False
