# Methodology: Neural Machine Translation Corpus Curation and Pre-processing

## 1. Data Acquisition
To establish a robust bilingual corpus for neural machine translation (NMT) evaluation and pre-training, we sourced parallel Bengali-English (`bn-en`) datasets from the OPUS (Open Parallel Corpus) collection. Specifically, the corpus integrates two distinct domains:
1. **OPUS-OpenSubtitles**: Sourced from the `opus100` subset via the Hugging Face Hub. This provides a large-scale, conversational domain dataset consisting of exactly 1,000,000 raw document pairs. 
2. **OPUS-Tatoeba**: Acquired directly via the OPUS server archives (`v2023-04-12` moses distribution). This constitutes a smaller, high-quality human-translated dataset comprising 5,572 raw sentence pairs, which introduces pristine grammatical structures to the corpus.

## 2. Structural Augmentation (Directional Flipping)
To train and evaluate the model symmetrically on both translation directions without runtime overhead, we applied structural data augmentation at download time. For every valid translation pair, two distinct records were generated and written in a structured token schema:
* **Bengali to English**: 
  `<|task_translate_bn_en|><|lang_bn|>{bn_text}<|lang_en|>{en_text}`
* **English to Bengali**: 
  `<|task_translate_en_bn|><|lang_en|>{en_text}<|lang_bn|>{bn_text}`

This deterministic mirroring inherently doubles the effective volume of the corpus, ensuring a mathematically balanced distribution of sequence objectives.

## 3. Heuristic Quality Filtering
During extraction, both corpora were subjected to rigorous deterministic filtering to discard spurious alignments, noisy data, and redundant structures:
1. **Length Constraints**: A minimum threshold of 3 words was enforced for both the Bengali and English sentences to eliminate disjointed 1-to-2 word phrases (e.g., "Yes.", "Hello."). Notably, **no upper boundary was enforced**; arbitrary-length documents were preserved to improve long-context translation fidelity.
2. **Alignment Ratio Constraint**: We mandated that the ratio of the English word count to the Bengali word count must reside strictly within the interval `[0.4, 2.5]`. Sequences violating this dynamic boundary were discarded on the assumption of alignment failure or heavy summarization.
3. **Deduplication**: An MD5 digest was computed for every incoming Bengali sequence. Exact topological duplicates were dropped globally across the processing stream, maximizing the linguistic diversity of the resultant dataset.

## 4. Orthographic Normalization
Following extraction and deduplication, the compiled `JSONL` files were subjected to a highly restrictive, language-isolated normalization phase. We utilized a custom Rust-based normalizer (`bn-normalize-rs`) engineered for computational efficiency and strict adherence to correct orthographic standards.

The normalizer selectively parsed the JSONL schema, isolating only the text bounded by `<|lang_bn|>` tokens. 
* Unicode strings were subjected to strict `NFC` normalization and whitespace collapsing.
* **Strict Drop Policy**: The pipeline was configured with a `drop_and_collect` policy. If the Rust normalizer detected corrupted bytes or inherently invalid lexical mappings within the Bengali segment, the *entire* document pair was structurally discarded to guarantee absolute corpus purity.

## 5. Tokenization and Sharding
The normalized NMT documents were linearly tokenized utilizing the project's pre-trained Hugging Face tokenizer (48,000 vocabulary size). Because the structured prompt schema (containing the specialized routing tokens like `<|task_translate_bn_en|>`) was injected at the download phase, no secondary prompt formatting was required at tokenization time.

The token streams were concatenated with standard EOS identifiers (`<|endoftext|>`) and subsequently packed into fixed, dense context windows of precisely `2048` tokens. The matrices were ultimately serialized as `uint16` NumPy arrays (`.npy`), with each shard encompassing exactly 100,000 sequences (204.8M tokens per shard), perfectly architected for highly efficient streaming to tensor cores during evaluation and pre-training phases.
