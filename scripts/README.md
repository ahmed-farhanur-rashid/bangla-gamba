# BanglaGamba Data Pipeline Scripts

Run in order:

1. `download/01a_download_titulm_cc.py`      — TituLM Common Crawl (4.5M docs)
2. `download/01b_download_titulm_romanized.py` — TituLM Romanized (~5GB)
3. `download/01c_download_wikipedia.py`        — Bengali Wikipedia (~90K articles)
4. `download/01d_download_banglanmt.py`        — BanglaNMT parallel pairs (2.75M)
5. `download/01e_download_banglishrev.py`      — BanglishRev reviews (~1M)
6. `pipeline/02_clean.py`                      — Clean, deduplicate, delete raw/
7. `pipeline/03_train_tokenizer.py`            — Train 48K BPE tokenizer, delete corpus
8. `pipeline/04_pretokenize.py`                — Pack into .npy shards, delete cleaned/
9. `pipeline/05_verify.py`                     — Sanity check shards before training

After step 8, set max_steps in configs/default_training.yaml (value printed by step 8).
Then: `python src/train.py --model configs/banglagamba_12l.yaml ...`
