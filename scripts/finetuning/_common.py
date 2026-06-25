"""Shared utilities for download scripts."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

RAW_DIR = Path("saved/data/raw")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def write_doc(f, text: str, source: str, source_type: str, language_region: str):
    doc = {
        "text": text,
        "source": source,
        "source_type": source_type,
        "language_region": language_region,
    }
    f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    try:
        from bnunicodenormalizer import Normalizer
        norm = Normalizer()
        text = norm.normalize(text)
        if isinstance(text, dict):
            text = text.get("normalized", text.get("text", ""))
    except (ImportError, Exception):
        pass
    text = " ".join(text.split())
    return text


def has_min_words(text: str, min_words: int = 20) -> bool:
    return len(text.split()) >= min_words
