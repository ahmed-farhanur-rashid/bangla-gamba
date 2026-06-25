# Corpus Preprocessing — BanglaGamba 200M
### Version 3 — ponytail reviewed

---

## Overview

Four data streams → one canonical JSONL schema → shared pipeline.

| Stream | Source | Pre-cleaned? | Work needed |
|---|---|---|---|
| Bangla monolingual | TituLLM | Yes — already deduped | NFC norm + write |
| Bangla monolingual | Wiki Bangla | Yes — clean | NFC norm + write |
| English monolingual | FineWeb-Edu | Yes — already deduped + quality filtered | NFC norm + write |
| NMT pairs | NLLB ben_Beng-eng_Latn | No — web mined | Full pipeline |
| NMT pairs | BanglaNMT (CSEBUET) | Mostly clean | Light filter + write |

---

## Canonical JSONL Schema

One document per line. Every stream, every source, same shape.

```json
{
  "text": "<special_token(s)>content here",
  "source": "titullm | wiki_bangla | fineweb_edu | nllb | banglanmt",
  "source_type": "web_bangla | web_english | parallel_bn_en",
  "language_region": "BD | EN | BN_EN_parallel",
  "word_count": 42
}
```

### Field rules

| Field | Exact allowed values | Effect in pipeline |
|---|---|---|
| `source_type` | `"web_bangla"`, `"web_english"`, `"parallel_bn_en"` | `02_clean.py` sets `min_words=10` for parallel, `20` for mono |
| `language_region` | `"BD"`, `"EN"`, `"BN_EN_parallel"` | `02_clean.py` runs ASCII filter only on `"BD_WB_mix"` — never set this on English docs |
| `word_count` | int | Computed on raw content, before special token prepend |

### Special token prefixes

Strings defined in `special_tokens.py`. Import from there — never hardcode.

| Stream | text field format |
|---|---|
| Bangla mono | `<\|lang_bn\|>{text}` |
| English mono | `<\|lang_en\|>{text}` |
| BN→EN | `<\|task_translate_bn_en\|><\|lang_bn\|>{bn}<\|lang_en\|>{en}` |
| EN→BN | `<\|task_translate_en_bn\|><\|lang_en\|>{en}<\|lang_bn\|>{bn}` |

---

## Schema Examples

```jsonl
{"text": "<|lang_bn|>এখানে বাংলা টেক্সট।", "source": "titullm", "source_type": "web_bangla", "language_region": "BD", "word_count": 847}
{"text": "<|lang_bn|>উইকিপিডিয়া নিবন্ধ।", "source": "wiki_bangla", "source_type": "web_bangla", "language_region": "BD", "word_count": 312}
{"text": "<|lang_en|>English text here.", "source": "fineweb_edu", "source_type": "web_english", "language_region": "EN", "word_count": 201}
{"text": "<|task_translate_bn_en|><|lang_bn|>আমি তোমাকে ভালোবাসি<|lang_en|>I love you", "source": "nllb", "source_type": "parallel_bn_en", "language_region": "BN_EN_parallel", "word_count": 8}
{"text": "<|task_translate_en_bn|><|lang_en|>I love you<|lang_bn|>আমি তোমাকে ভালোবাসি", "source": "nllb", "source_type": "parallel_bn_en", "language_region": "BN_EN_parallel", "word_count": 8}
```

Each NMT pair produces **two lines** — one per direction.

---

## Shared Utilities

```python
import hashlib
import json
import re
import unicodedata
from special_tokens import (
    LANG_BN, LANG_EN,
    TASK_BN_EN, TASK_EN_BN,
)
# Add to special_tokens.py if not present:
# LANG_BN    = "<|lang_bn|>"
# LANG_EN    = "<|lang_en|>"
# TASK_BN_EN = "<|task_translate_bn_en|>"
# TASK_EN_BN = "<|task_translate_en_bn|>"

BANGLA_RE = re.compile(r'[\u0980-\u09FF]')


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def wc(text: str) -> int:
    return len(text.split())


def write_doc(f_out, text, source, source_type, language_region, word_count):
    f_out.write(json.dumps({
        "text": text,
        "source": source,
        "source_type": source_type,
        "language_region": language_region,
        "word_count": word_count,
    }, ensure_ascii=False) + "\n")


# Shared NMT dedup state — reset between full pipeline reruns
# ponytail: global set, fine for one-shot script; replace with BloomFilter if 62M pairs OOM
seen_bn: set[bytes] = set()


def is_dup_nmt(bn: str) -> bool:
    h = hashlib.md5(bn.encode()).digest()
    if h in seen_bn:
        return True
    seen_bn.add(h)
    return False
```

---

## Step 1 — Unicode Normalization

Apply NFC as the **absolute first step** on every stream before any filter.

**NFC not NFKC:** NFKC collapses compatibility characters and can alter Bangla
conjunct forms in legacy-converted text. NFC is safe.

**Why it matters for dedup:** Without NFC, identical documents with different
Unicode encodings get different MD5/SHA-256 hashes and slip through exact dedup.
This is common in Bangla web text due to inconsistent keyboard encoding.

---

## Step 2 — Bangla Monolingual (TituLLM + Wiki Bangla)

TituLLM is already deduplicated (arXiv:2502.11187). Wiki Bangla is clean.
No fuzzy dedup. Exact dedup only at the cross-source merge step.

```python
def clean_bangla(text: str) -> tuple[str, int] | None:
    text = nfc(text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    n = wc(text)
    if n < 50:
        return None
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return None
    if sum(1 for c in chars if BANGLA_RE.match(c)) / len(chars) < 0.70:
        return None
    return text, n


def process_bangla(raw_docs, f_out, source: str):
    """Works for both titullm and wiki_bangla — pass source name."""
    for raw in raw_docs:
        result = clean_bangla(raw)
        if result is None:
            continue
        text, n = result
        write_doc(f_out, f"{LANG_BN}{text}", source, "web_bangla", "BD", n)
```

Call as:
```python
process_bangla(titullm_docs, f_out, source="titullm")
process_bangla(wiki_docs,    f_out, source="wiki_bangla")
```

---

## Step 3 — English Monolingual (FineWeb-Edu)

Already quality filtered and MinHash-deduplicated by HuggingFace.
NFC normalize and write. Nothing else.

```python
def process_english(raw_docs, f_out):
    for raw in raw_docs:
        text = nfc(raw).strip()
        if not text:
            continue
        write_doc(f_out, f"{LANG_EN}{text}", "fineweb_edu", "web_english", "EN", wc(text))
```

---

## Step 4 — NMT Pairs (NLLB + BanglaNMT)

### Filters (applied to both sources)

```python
def length_ok(bn: str, en: str) -> bool:
    b, e = wc(bn), wc(en)
    return 3 <= b <= 150 and 3 <= e <= 150


def ratio_ok(bn: str, en: str) -> bool:
    # Bangla is morphologically rich: one BN word ~ 1.5-2 EN words
    # Asymmetric window accounts for natural skew above 1.0
    return 0.4 <= wc(en) / wc(bn) <= 2.5
```

### NLLB pipeline

```python
def process_nmt(pairs, f_out, source: str):
    """
    pairs:  iterable of (bn_str, en_str)
    source: "nllb" or "banglanmt"

    Run NLLB first, then BanglaNMT — they share seen_bn
    so cross-source duplicates are caught automatically.
    """
    kept = dropped = 0

    for bn_raw, en_raw in pairs:
        bn = nfc(bn_raw)
        en = nfc(en_raw)

        if not length_ok(bn, en) or not ratio_ok(bn, en) or is_dup_nmt(bn):
            dropped += 1
            continue

        n = wc(bn) + wc(en)
        write_doc(f_out, f"{TASK_BN_EN}{LANG_BN}{bn}{LANG_EN}{en}", source, "parallel_bn_en", "BN_EN_parallel", n)
        write_doc(f_out, f"{TASK_EN_BN}{LANG_EN}{en}{LANG_BN}{bn}", source, "parallel_bn_en", "BN_EN_parallel", n)
        kept += 1

    print(f"[{source}] kept={kept:,} dropped={dropped:,} ({dropped / max(kept+dropped,1)*100:.1f}% removed)")
```

Call order matters — NLLB first to populate `seen_bn`, then BanglaNMT:
```python
process_nmt(nllb_pairs,      nmt_out, source="nllb")
process_nmt(banglanmt_pairs, nmt_out, source="banglanmt")
```

---

## Step 5 — What NOT to do

| Don't | Why |
|---|---|
| Run MinHash on TituLLM or NLLB | TituLLM already deduped. NLLB pairs are exact dupes or not — fuzzy is wrong tool. |
| Use NFKC | Too aggressive for Bangla conjuncts. NFC only. |
| Use Python `hash()` for dedup | Not stable across runs. Use `hashlib.md5`. |
| Prepend lang tokens inside cleaners | Cleaners return plain text + word count. Schema assembly at write time only. |
| Set `language_region="BD_WB_mix"` on English | Triggers ASCII filter that rejects high-ASCII text — correct for Bangla, wrong for English. |
| Process BanglaNMT before NLLB | They share `seen_bn` — order determines which source "wins" on duplicates. NLLB first. |

---

## Step 6 — Processing Order

```
1. process_bangla(titullm_docs, bangla_out, "titullm")
2. process_bangla(wiki_docs,    bangla_out, "wiki_bangla")
3. process_english(fineweb_docs, english_out)
4. process_nmt(nllb_pairs,      nmt_out, "nllb")        ← populates seen_bn
5. process_nmt(banglanmt_pairs, nmt_out, "banglanmt")   ← deduped against NLLB

Cross-source exact dedup (after all streams):
6. python scripts/pipeline/02b_dedup.py --exact-only --no-delete
   (catches any doc that appears in both TituLLM and Wiki, etc.)

Shard assembly:
7. shards/bangla/   ← bangla_out (titullm + wiki_bangla interleaved)
   shards/english/  ← english_out
   shards/nmt/      ← nmt_out (both directions already interleaved)
8. Pack each directory into 2048-token NPY sequences
```

---

## Step 7 — Expected Output Sizes

| Stream | Raw | After filtering | Est. tokens |
|---|---|---|---|
| TituLLM | ~5.2B words | ~5.0B words | ~7.28B |
| Wiki Bangla | ~51M words | ~48M words | ~71M |
| FineWeb-Edu | streamed to ~700M words | ~700M words | ~1.0B |
| NLLB (both dirs) | 62M pairs | ~35–45M pairs × 2 | ~1.0–1.2B |
| BanglaNMT (both dirs) | 2.75M pairs | ~2.5M pairs × 2 | ~85M |

**Total: ~9.4–9.7B tokens**

---

## Step 8 — Final Composition Check

Before tokenizer training, verify effective language exposure:

| Language | Sources contributing | Target |
|---|---|---|
| Bangla | TituLLM + Wiki + BN-side of NMT | ~85% |
| English | FineWeb-Edu + EN-side of NMT | ~15% |

If NMT is over-represented, downsample at the dataloader sampling weight —
do not re-filter the JSONL.
