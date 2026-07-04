# Data Pipeline

> **Target:** 8–10B tokens · 85% Bangla / 15% English
> **Sources:** TituLLM CC · Sangraha Verified · FineWeb-Edu · BanglaNMT · NLLB
> **Tokenizer corpus:** 250M words (85% Bangla / 15% English)

---

## Directory Layout

```
saved/data/
  raw/              ← downloaded JSONL files (untouched after download)
  deduped/          ← hash-deduped, no normalization yet
  cleaned/          ← normalized final output
  tokenizer_set/    ← sampled corpus for tokenizer training
  pretokenized/     ← .npy shards per source type
  logs/             ← normalization failure logs
```

---

## Prerequisites

### Install the Rust Normalizer

The pipeline uses [bn-normalizer-rs](https://github.com/ahmed-farhanur-rashid/bn-normalizer-rs) for fast Bangla Unicode normalization (~26x faster than the Python original). Requires a Rust toolchain.

```bash
# Install Rust (if not already)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone and install
git clone https://github.com/ahmed-farhanur-rashid/bn-normalizer-rs.git
cd bn-normalizer-rs
pip install maturin
maturin develop --release
```

### Prevent OOM Errors

Run these **before Step 3 (pretokenize)**, or your process will be killed under memory pressure.

```bash
# Stop systemd-oomd
sudo systemctl stop systemd-oomd
sudo systemctl stop systemd-oomd.socket systemd-oomd.service

# Add swap (recommended: 120 GB with 32 GB RAM)
sudo swapoff -a
sudo fallocate -l 120G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Step 1 — Download Datasets

All downloaders write JSONL to `saved/data/raw/` with this schema:

```json
{"text": "<|lang_bn|>...", "source": "titullm", "source_type": "web_bangla", "language_region": "BD", "word_count": 346}
```

Every downloader applies:
- **NFC normalization** (`unicodedata.normalize("NFC", text)`) at download time
- **Paragraph collapse** — 3+ consecutive newlines → 2 (preserves paragraph breaks)
- **Minimum word filter** — docs with < 20 words are dropped
- **Language token prefix** — `<|lang_bn|>` for Bangla, `<|lang_en|>` for English
- **Resume support** — re-run skips already-downloaded lines

### Bangla Sources

```bash
python pretrain-corpus-pipeline/downloaders/01a_download_titulm_cc.py
python pretrain-corpus-pipeline/downloaders/01b_download_wikipedia_bn.py
python pretrain-corpus-pipeline/downloaders/01c_download_sangraha_bn.py
```

| Script | Source | Output | Notes |
|---|---|---|---|
| `01a` | TituLM Common Crawl (`hishab/titulm-bangla-corpus`) | `raw/titullm_cc.jsonl` | ~15M docs |
| `01b` | Bengali Wikipedia (`wikimedia/wikipedia`, 20231101.bn) | `raw/wiki_bangla.jsonl` | ~143K docs |
| `01c` | Sangraha Verified (`ai4bharat/sangraha`, verified/ben) | `raw/sangraha_verified_bn.jsonl` | ~5.5M docs |

> **Warning:** Do NOT use split `asm` (Assamese) in Sangraha — `datasets.load_dataset` silently succeeds on either.

### English Source

```bash
python pretrain-corpus-pipeline/downloaders/02a_download_english.py --word-budget 2_000_000_000
```

| Script | Source | Output | Notes |
|---|---|---|---|
| `02a` | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`) | `cleaned/english.jsonl` | ~2B words, goes directly to cleaned/ |

English includes an additional content filter (adult/spam pattern matching) beyond the standard preprocessing.

### NMT Sources

```bash
python pretrain-corpus-pipeline/downloaders/03a_download_banglanmt.py
python pretrain-corpus-pipeline/downloaders/03b_download_nllb.py
```

| Script | Source | Output | Notes |
|---|---|---|---|
| `03a` | BanglaNMT (`csebuetnlp/BanglaNMT`) | `raw/banglanmt.jsonl` | Both translation directions |
| `03b` | NLLB (AllenAI bucket, LASER-filtered) | `raw/nllb.jsonl` | Score ≥ 1.06 |

NMT pairs are written **twice** per pair (one per direction) with task tokens:
```
<|task_translate_bn_en|><|lang_bn|>...<|lang_en|>...
<|task_translate_en_bn|><|lang_en|>...<|lang_bn|>...
```

Both sides are whitespace-collapsed (no paragraph breaks). Pairs are filtered by length (3–150 words per side) and translation ratio (0.4–2.5).

---

## Step 2 — Dedup & Quality Check

### NMT Dedup

```bash
python pretrain-corpus-pipeline/01a_dedup_nmt.py
```

- **Input:** `raw/nllb.jsonl` + `raw/banglanmt.jsonl`
- **Output:** `cleaned/nmt.jsonl`
- **Method:** MD5 hash dedup on full text field. NLLB processed first (higher priority).

### Bangla Base Dedup (Wiki + TituLLM)

```bash
python pretrain-corpus-pipeline/01b_dedup_mono_bn.py
```

- **Input:** `raw/wiki_bangla.jsonl` + `raw/titullm_cc.jsonl`
- **Output:** `deduped/bangla_deduped.jsonl`
- **Method:** SHA-256 hash of NFC-normalized + whitespace-collapsed text. Wiki processed first (higher priority).
- **No normalization applied** — just hash dedup. The Rust normalizer runs in Step 3.

### Sangraha Dedup Against Base

```bash
python pretrain-corpus-pipeline/01c_dedup_sangraha.py
python pretrain-corpus-pipeline/01c_dedup_sangraha.py --max-words 2_000_000_000  # optional cap
```

- **Input:** `deduped/bangla_deduped.jsonl` (read-only, builds hash set) + `raw/sangraha_verified_bn.jsonl`
- **Output:** `deduped/sangraha_deduped.jsonl`
- **Method:** Loads all base hashes into memory, streams Sangraha, drops matches against base + within-Sangraha duplicates.
- **Flags:** `--max-words N`, `--max-docs N`, `--dry-run`

---

## Step 3 — Normalize (Rust)

```bash
# Bangla (wiki + titullm)
python pretrain-corpus-pipeline/01d_bn_normalize.py \
  --input saved/data/deduped/bangla_deduped.jsonl \
  --output saved/data/cleaned/bangla.jsonl \
  --none-policy drop_and_collect

# Sangraha (West Bengali — kept separate for ratio control)
python pretrain-corpus-pipeline/01d_bn_normalize.py \
  --input saved/data/deduped/sangraha_deduped.jsonl \
  --output saved/data/cleaned/sangraha.jsonl \
  --none-policy drop_and_collect
```

- **Input:** `deduped/*.jsonl`
- **Output:** `cleaned/*.jsonl` + `saved/logs/*_norm_failures.jsonl`
- **Method:** Calls `bn_normalize_rs.normalize_sentence()` (Rust, ~26x faster than Python). Applies Bangla-specific normalization: broken diacritics, nukta composition, virama cleanup, conjunct normalization, complex root validation, and more.
- **`--none-policy drop_and_collect`:** Drops invalid tokens from text AND logs them to `saved/logs/`. Invalid tokens are standalone diacritics from OCR/scraping artifacts — garbage that would hurt training.
- **`--allow-english` (default: True):** English characters treated as valid in word normalization. Keeps mixed-script text intact.

---

## Step 4 — Sample & Train Tokenizer

### Sample Corpus

```bash
python src/tokenizer/tokenizer_sampler.py \
  --total-words 250_000_000 \
  --ratio 0.85
```

- **Input:** `cleaned/bangla.jsonl` + `cleaned/english.jsonl`
- **Output:** `tokenizer_set/corpus.jsonl` (shuffled mix)
- **Method:** Streams both files, samples until word budget is met (212.5M Bangla + 37.5M English), then weighted random merge. Outputs only `{"text": ...}` (strips schema metadata).

### Train Tokenizer

```bash
python3 -m src.tokenizer.train_tokenizer \
  --input saved/data/tokenizer_set/corpus.jsonl \
  --jsonl \
  --output-dir saved/tokenizer \
  --input-sentence-size 0 \
  --max-sentence-length 65536 \
  --num-threads 4 2>&1 | tee saved/logs/tokenizer_train.log
```

### Wrap & Evaluate

```bash
# Wrap SPM into HF format
python -m src.tokenizer.wrapper \
  --spm-model saved/tokenizer/banglagamba_tokenizer.model \
  --output-dir saved/tokenizer/hf \
  --test

# Fix tokenizer_class
sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' \
  saved/tokenizer/hf/tokenizer_config.json

# Fix model_max_length
sed -i 's/"model_max_length": [0-9]*/"model_max_length": 2048/' \
  saved/tokenizer/hf/tokenizer_config.json

# Evaluate
python tests/evaluate_tokenizer.py
```

---

## Step 5 — Pretokenize & Pack Sequences

```bash
python scripts/pretokenizer.py --source bangla
python scripts/pretokenizer.py --source sangraha
python scripts/pretokenizer.py --source english
python scripts/pretokenizer.py --source nmt
```

Or all at once:

```bash
python scripts/pipeline/pretokenizer.py
```

- **Input:** `cleaned/{bangla,sangraha,english,nmt}.jsonl`
- **Output:** `pretokenized/{bangla,sangraha,english,nmt}/train/shard_*.npy`
- **Method:** Tokenizes each doc with the trained HF tokenizer, appends EOS, packs into 2048-token sequences, writes as NumPy `uint16` arrays.
- **Shard size:** 204.8M tokens (2048 × 100K rows)
- **Sangraha is separate** — kept apart from Bangla for training-time ratio control. Some shards can be held out for eval.

After pretokenization, calculate training steps:

```bash
# Approximate steps
# tokens_per_step = batch_size * gradient_accumulation * seq_len
# e.g., 4 * 64 * 2048 = 524,288 tokens/step
```

Set `max_steps` in `configs/default_training.yaml` accordingly.
