# Data Pipeline

> **Target:** 8–10B tokens · 85% Bangla / 15% English  
> **Sources:** TituLLM CC · Sangraha Verified · FineWeb-Edu · BanglaNMT · NLLB

---

## Step 1 — Download Datasets

```bash
python scripts/downloaders/01a_download_titulm_cc.py
python scripts/downloaders/01b_download_wikipedia_bn.py
python scripts/downloaders/01c_download_sangraha_bn.py
python scripts/downloaders/02a_download_english.py --word-budget 2_000_000_000
python scripts/downloaders/03a_download_banglanmt.py
python scripts/downloaders/03b_download_nllb.py
```

---

## Step 2 — Dedup & Quality Check

```bash
# NMT dedup (NLLB + BanglaNMT)
python scripts/pipeline/01a_dedup_nmt.py

# Bangla base dedup (wiki + titullm) → deduped/
python scripts/pipeline/01b_dedup_mono_bn.py

# Sangraha dedup against base → deduped/
python scripts/pipeline/01c_dedup_sangraha.py
python scripts/pipeline/01c_dedup_sangraha.py --max-words 2_000_000_000  # optional cap

# Normalize deduped → cleaned/
python scripts/pipeline/01d_bn_normalize.py
```

---

## Preparation — Prevent OOM Errors

> Run these **before Step 3**, or your process will be killed under memory pressure.

### Stop systemd-oomd

```bash
sudo systemctl stop systemd-oomd
sudo systemctl stop systemd-oomd.socket systemd-oomd.service
systemctl status systemd-oomd
```

### Add Swap Space

Recommended: **120 GB swap** with 32 GB RAM. Adjust proportionally for your setup.

```bash
# Check current swap
swapon --show
free -h

# Create new swapfile
sudo swapoff -a
sudo rm -f /swapfile
sudo fallocate -l 120G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
swapon --show
free -h
```

---

## Step 3 — Sample & Train Tokenizer

```bash
python scripts/pipeline/02_tokenizer_sampler.py \
    --total-words 500_000_000 \
    --ratio 0.85 && \
python3 -m src.tokenizer.train_tokenizer \
    --input saved/data/tokenizer_set/corpus.jsonl \
    --jsonl \
    --output-dir saved/tokenizer \
    --input-sentence-size 0 \
    --max-sentence-length 65536 \
    --num-threads 4 2>&1 | tee saved/logs/tokenizer_train.log
```

---

## Step 4 — Post-Tokenizer Setup

### 1. Wrap SPM model into HF format

```bash
python -m src.tokenizer.wrapper \
  --spm-model saved/tokenizer/banglagamba_tokenizer.model \
  --output-dir saved/tokenizer/hf
```

### 2. Sanity check (decode, special tokens, chat template)

```bash
python scripts/util/evaluate_tokenizer.py --sanity --skip-references
```

### 3. Full evaluation against reference tokenizers

```bash
python scripts/util/evaluate_tokenizer.py
```

### 4. Wrap + sanity in one shot

```bash
python -m src.tokenizer.wrapper \
  --spm-model saved/tokenizer/banglagamba_tokenizer.model \
  --output-dir saved/tokenizer/hf \
  --test
```

### 5. Apply fixes

**Fix `tokenizer_class`** — `TokenizersBackend` should be `PreTrainedTokenizerFast`, otherwise `AutoTokenizer.from_pretrained()` may fail or warn.

```bash
sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' \
  saved/tokenizer/hf/tokenizer_config.json
```

**Fix `model_max_length`** — the default HF sentinel value is nonsensical. Set it to your actual context length.

```bash
sed -i 's/"model_max_length": [0-9]*/"model_max_length": 2048/' \
  saved/tokenizer/hf/tokenizer_config.json
```

### `evaluate_tokenizer.py` flags

| Flag | Description |
|---|---|
| `--skip-references` | Skip downloading reference models |
| `--categories bangla_formal english` | Run specific categories only |
| `--sample-size 5000` | Use a smaller corpus sample |
| `--no-corpus` | Skip corpus tests; use curated sentences only |

---

## Step 5 — Pretokenize & Pack Sequences

Run each source separately:

```bash
python scripts/pipeline/03_pretokenize.py --source bangla
python scripts/pipeline/03_pretokenize.py --source sangraha
python scripts/pipeline/03_pretokenize.py --source english
python scripts/pipeline/03_pretokenize.py --source nmt
```

Or all at once:

```bash
python scripts/pipeline/03_pretokenize.py
```