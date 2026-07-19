# Dataset Packaging and Upload Guide

This document contains the exact CLI commands required to pack your large `.jsonl` datasets into Hugging Face-compatible `.parquet` shards (with a strict row-group size limit to prevent `TooBigContentError`) and seamlessly upload them to your repository.

**Important:** Make sure you are running these commands from the root of your project directory (`/home/farhan/my-projects/bangla-gamba`).

---

## 1. Bangla Corpus (`bangla.jsonl`)

**Pack:**
```bash
python pretrain-corpus-pipeline/util/pack_to_parquet.py \
    --files bangla.jsonl \
    --output-dir temp/bangla_corpus \
    --row-group-size 50000
```

**Upload:**
```bash
python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir temp/bangla_corpus \
    --path-in-repo bangla_corpus
```

---

## 2. Fine Web Edu / English (`english.jsonl`)

**Pack:**
```bash
python pretrain-corpus-pipeline/util/pack_to_parquet.py \
    --files english.jsonl \
    --output-dir temp/fine_web_edu \
    --row-group-size 50000
```

**Upload:**
```bash
python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir temp/fine_web_edu \
    --path-in-repo fine_web_edu
```

---

## 3. Sangraha (`sangraha.jsonl`)

**Pack:**
```bash
python pretrain-corpus-pipeline/util/pack_to_parquet.py \
    --files sangraha.jsonl \
    --output-dir temp/sangraha \
    --row-group-size 50000
```

**Upload:**
```bash
python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir temp/sangraha \
    --path-in-repo sangraha
```

---

## 4. NMT Datasets (Optional, already uploaded)
*(Included for completeness if you ever need to re-process translation data)*

**Pack:**
```bash
python pretrain-corpus-pipeline/util/pack_to_parquet.py \
    --files nmt.jsonl opus_nmt.jsonl \
    --output-dir temp/nmt \
    --row-group-size 50000
```

**Upload:**
```bash
python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir temp/nmt \
    --path-in-repo nllb_nmt
```
