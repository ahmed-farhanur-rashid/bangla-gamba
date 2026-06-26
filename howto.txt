============================================================
  Data Pipeline
============================================================
  Target: 8-10B of total tokens maintaining 85:15 Bng:Eng ratio
  Sources: Titullm CC + Fineweb-edu + BanglaNMT + NLLB


============================================================
  STEP 1: Download Datasets
============================================================
  python scripts/downloaders/01a_download_titulm_cc.py
  python scripts/downloaders/01b_download_wikipedia.py
  python scripts/downloaders/02a_download_english.py --word-budget 2_000_000_000
  python scripts/downloaders/03a_download_banglanmt.py
  python scripts/downloaders/03b_download_nllb.py


============================================================
  STEP 2: Dedup & check quality
============================================================
  python scripts/pipeline/01a_dedup_nmt.py
  python scripts/pipeline/01b_dedup_mono.py


============================================================
  Preperation: Run these, otherwise you'll hit OOM Error 
============================================================

This stops systemd-oomd (Systemd Out-Of-Memory Daemon)
Without this, your process will be killed due to "memory pressure"

  sudo systemctl stop systemd-oomd 
  sudo systemctl stop systemd-oomd.socket systemd-oomd.service
  systemctl status systemd-oomd

Add a 120G swap to device (provided that you have 32G ram)
Add more swap if less and less if you have more ram

  swapon --show
  free -h

  sudo swapoff -a
  sudo rm -f /swapfile
  sudo fallocate -l 120G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile

  swapon --show
  free -h

============================================================
  STEP 3: Sample Tokenizer Training Set & Train tokenizer 
============================================================
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


===================================================
  Step 4: Post Tokenizer Training            
===================================================
  1. Wrap the SPM model into HF format

  python -m src.tokenizer.wrapper \
    --spm-model saved/tokenizer/banglagamba_tokenizer.model \
    --output-dir saved/tokenizer/hf

  2. Quick sanity check (decode, special tokens, chat template)

  python scripts/util/evaluate_tokenizer.py \
    --sanity --skip-references

  3. Full evaluation against reference tokenizers

  python scripts/util/evaluate_tokenizer.py

  4. Wrap + sanity in one shot

  python -m src.tokenizer.wrapper \
    --spm-model saved/tokenizer/banglagamba_tokenizer.model \
    --output-dir saved/tokenizer/hf \
    --test

  5. Apply a few fixes

  Problem:
  tokenizer_class": "TokenizersBackend" is wrong.
  It should be "PreTrainedTokenizerFast", otherwise HF's AutoTokenizer.from_pretrained() may fail or warn.

  fix:
  sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' saved/tokenizer/hf/tokenizer_config.json

  Problem:
  "model_max_length": 1000000000000000019884624838656 — that's HF's default sentinel for "not set". 
  You'll want to set this to your actual context length when you define your model architecture, 
  otherwise HF will warn on long sequences

  fix:
  sed -i 's/"model_max_length": [0-9]*/"model_max_length": 2048/' saved/tokenizer/hf/tokenizer_config.json

  Optional flags for evaluate_tokenizer:
    --skip-references          skip downloading reference models
    --categories bangla_formal english   run specific categories only
    --sample-size 5000         smaller corpus sample
    --no-corpus                skip corpus tests, curated sentences only


===================================================
  Step 5: Pretokenization and Sequence Packing           
===================================================
  Run all three sources separately

  python scripts/pipeline/03_pretokenize.py --source bangla
  python scripts/pipeline/03_pretokenize.py --source english
  python scripts/pipeline/03_pretokenize.py --source nmt

  Or all at once

  python scripts/pipeline/03_pretokenize.py