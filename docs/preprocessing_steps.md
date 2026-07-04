# Corpus Preprocessing — BanglaGamba 200M

---

## Overview

Five data streams → one canonical JSONL schema → shared pipeline.

| Stream | Source | Pre-cleaned? | Work needed |
|---|---|---|---|
| Bangla monolingual | TituLLM | Yes — already deduped | Dedup + NFC norm (drop_and_collect) |
| Bangla monolingual | Wiki Bangla | Yes — clean | Dedup + NFC norm (drop_and_collect) |
| Bangla monolingual | Sangraha | Mostly clean | Dedup against base + NFC norm |
| English monolingual | FineWeb-Edu | Yes — already deduped + quality filtered | NFC norm |
| NMT pairs | NLLB ben_Beng-eng_Latn | No — web mined | Dedup + Full pipeline |
| NMT pairs | BanglaNMT (CSEBUET) | Mostly clean | Dedup against NLLB + filter |

---

## Canonical JSONL Schema

One document per line. Every stream, every source, same shape.

```json
{
  "text": "<special_token(s)>content here",
  "source": "titullm | wiki_bangla | sangraha | fineweb_edu | nllb | banglanmt"
}
```

### Special token prefixes

Strings defined in the tokenizer vocabulary.

| Stream | text field format |
|---|---|
| Bangla mono | `<|lang_bn|>{text}` |
| English mono | `<|lang_en|>{text}` |
| BN→EN | `<|task_translate_bn_en|><|lang_bn|>{bn}<|lang_en|>{en}` |
| EN→BN | `<|task_translate_en_bn|><|lang_en|>{en}<|lang_bn|>{bn}` |

---

## Step 1 — Cross-Source Deduplication

Deduplication occurs *before* normalization, separating distinct stages:
- **NMT Dedup**: `01a_dedup_nmt.py` removes intersection between NLLB and BanglaNMT.
- **Bangla Base Dedup**: `01b_dedup_mono_bn.py` removes intersection between Wiki Bangla and TituLLM.
- **Sangraha Dedup**: `01c_dedup_sangraha.py` removes Sangraha documents already present in the Base corpus.

Uses exact matching with `hashlib.md5`.

---

## Step 2 — Unicode Normalization (Rust)

Apply NFC normalization using the high-performance Rust library `bn_normalize_rs`.

**NFC not NFKC:** NFKC collapses compatibility characters and can alter Bangla
conjunct forms in legacy-converted text. NFC is safe.

**Drop and Collect Policy:** We use the `--none-policy drop_and_collect` argument in `01d_bn_normalize.py`.
Tokens that fail valid Bangla unicode validation are discarded from the document, but saved to a `_failures.jsonl` log for offline analysis.

---

## Step 3 — NMT Pair Filtering

NMT pairs undergo length and ratio filtering:
1. `3 <= words <= 150` for both sides.
2. `0.4 <= wc(en) / wc(bn) <= 2.5` to account for morphological richness of Bangla.

---

## Step 4 — What NOT to do

| Don't | Why |
|---|---|
| Run MinHash on TituLLM or NLLB | TituLLM already deduped. NLLB pairs are exact dupes or not — fuzzy is wrong tool. |
| Use NFKC | Too aggressive for Bangla conjuncts. NFC only. |
| Use Python `hash()` for dedup | Not stable across runs. Use `hashlib.md5`. |
| Process BanglaNMT before NLLB | They share a hash set. NLLB first. |

---

## Step 5 — Final Composition

1. `pretrain-corpus-pipeline/` scripts handle download, dedup, and normalize.
2. `src/tokenizer/tokenizer_sampler.py` creates a 500M word representative sample at an 85:15 (BN:EN) ratio.
3. Train SentencePiece tokenizer.
4. `scripts/pretokenizer.py` packs all sequences into `uint16` `.npy` shards.
5. `tests/verify_pretokenized_shards.py` asserts sequence length and vocab boundaries before training begins.
