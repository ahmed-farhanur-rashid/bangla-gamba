# Methodology: BanglaGamba 200M

## 1. Model Architecture

BanglaGamba is a 199M parameter hybrid language model that interleaves Mamba-3 selective state space blocks with Grouped Query Attention (GQA) blocks in a 1:1 pattern. The architecture targets efficient sequence modeling on consumer hardware (RTX 4070 Super, 12 GB VRAM) while maintaining competitive representational capacity.

### 1.1 Architectural Overview

The model consists of 12 blocks (6 Mamba-3, 6 GQA) with a fixed sequence length of 2048 tokens. Each block contains a mixer sublayer (either Mamba-3 or GQA) followed by a SwiGLU feedforward sublayer, both pre-normed with RMSNorm and connected via residual connections.

**Table 1: Model hyperparameters.**

| Parameter | Value | Rationale |
|---|---|---|
| d_model | 1024 | Balances capacity with VRAM constraints |
| n_layers | 12 | 6 interleaved pairs |
| n_heads (GQA) | 16 | Standard for this scale |
| n_kv_heads (GQA) | 4 | Grouped query attention (4x KV compression) |
| d_head | 64 | Standard head dimension |
| d_ff (SwiGLU) | 2560 | 2/3 × 4 × d_model, rounded to nearest 256 |
| Vocab size | 48,000 | 146 special tokens + 47,854 learned subwords |
| Sequence length | 2048 | Fixed throughout training |
| Mamba d_state | 128 | Mamba-2/3 state dimension |
| Mamba expand | 2 | d_inner = 2048 |
| Mamba headdim | 64 | 32 internal SSM heads |
| Mamba chunk_size | 64 | Optimized for RTX 4070 Super shared memory |
| RoPE base | 10000.0 | Standard rotary position embedding |
| Tie embeddings | true | Saves ~49M parameters (LM head shares embedding weights) |

### 1.2 Layer Pattern

The layer stack follows a strict alternating pattern: Mamba-3 blocks occupy even indices (0, 2, 4, 6, 8, 10) and GQA blocks occupy odd indices (1, 3, 5, 7, 9, 11), with Mamba-3 as the first layer and GQA as the last. This 1:1 interleaving follows the Jamba/Lieber et al. hybrid architecture design, where linear-complexity SSM blocks handle local pattern extraction and attention blocks enable global context aggregation [10].

### 1.3 Stability Mechanisms

Three stability mechanisms are employed:

1. **QK-Norm**: Per-head RMSNorm is applied to Q and K projections before RoPE. This prevents logit magnitude drift in attention layers, which is critical for stable training with the Muon optimizer.

2. **Residual init scaling**: Output projections of all mixer and FFN sublayers (out_proj for Mamba, o_proj for GQA, down_proj for SwiGLU) are scaled by $1/\sqrt{2 \cdot n_{\text{layers}}}$. This prevents activation variance from growing with depth during early training.

3. **Z-loss**: A logit magnitude penalty with weight $10^{-4}$ is applied to suppress late-training loss spikes.

### 1.4 Optimizer

A hybrid optimizer configuration is used:

- **Muon** (momentum 0.95, Nesterov, 5 Newton-Schulz iterations, lr=0.02): Applied to all 2D dense matrix weights (~150M parameters). Muon provides superior optimization geometry for weight matrices compared to Adam-based methods.
- **AdamW** ($\beta_1=0.9$, $\beta_2=0.95$, lr=$3 \times 10^{-4}$, fused): Applied to everything else (~49M parameters — embeddings, norms, 1D parameters).

The effective batch size is $4 \times 64 \times 2048 = 524,288$ tokens per step (batch_size=4, gradient_accumulation=64). Training uses BF16 with FP32 optimizer states. Gradient checkpointing and torch.compile (reduce-overhead mode) are enabled.

---

## 2. Training Data

### 2.1 Data Sources

Four data streams are collected, each with different preprocessing requirements based on their source quality.

**Table 2: Data sources.**

| Stream | Source | Language | Volume (est.) | Pre-cleaned? | Work needed |
|---|---|---|---|---|---|
| Bangla monolingual | TituLLM (Common Crawl) [16] | BN | ~5B words | Yes — already deduped | NFC norm + source filters |
| Bangla monolingual | Wikipedia BN | BN | ~50M words | Yes — clean | NFC norm + source filters |
| English monolingual | FineWeb-Edu [13] | EN | ~1-2B words | Yes — quality filtered + deduped | NFC norm only |
| NMT pairs | NLLB ben_Beng-eng_Latn [8] | BN↔EN | ~62M pairs | No — web mined | Full filter pipeline |
| NMT pairs | BanglaNMT CSEBUET | BN↔EN | ~2.75M pairs | Mostly clean | Light filter + cross-source dedup |

**TituLLM** is a large-scale Bangla web corpus derived from Common Crawl, already deduplicated by the TituLLM team [16]. It serves as the primary Bangla monolingual backbone.

**FineWeb-Edu** is a quality-filtered English web corpus released by HuggingFace [13]. It has already undergone MinHash deduplication and educational quality scoring, so no further filtering is applied beyond NFC normalization.

**NLLB** (No Language Left Behind) provides web-mined parallel bitext for Bangla-English [8]. These pairs require the most preprocessing due to noisy web-mined data.

**BanglaNMT** provides curated parallel pairs from CSEBUET, serving as higher-quality supplementary NMT data.

### 2.2 Data Composition and Ratio

The target composition is **85% Bangla, 15% English** (by word count). This ratio is chosen to:

1. Prioritize Bangla language understanding while maintaining sufficient English exposure for cross-lingual transfer
2. Ensure the tokenizer allocates vocab capacity proportional to actual training exposure
3. Align with the observation that Bangla is morphologically richer than English — one Bangla word typically corresponds to 1.5–2 English words [preprocessing_steps.md]

The ratio is enforced at two stages:
- **Tokenizer training**: 85:15 BN:EN sampling when constructing the tokenizer training corpus
- **Pretokenization**: Separate shard directories per source type enable dataloader-level sampling weight adjustments

---

## 3. Preprocessing Pipeline

The preprocessing pipeline converts raw downloads into pretokenized `.npy` shards ready for training. The pipeline operates in six sequential stages.

### 3.1 Stage 1: Download and Source-Level Filtering

Each download script writes documents in a canonical JSONL schema:

```json
{
  "text": "<special_token(s)>content",
  "source": "titullm | wiki_bangla | fineweb_edu | nllb | banglanmt",
  "source_type": "web_bangla | web_english | parallel_bn_en",
  "language_region": "BD | EN | BN_EN_parallel",
  "word_count": 42
}
```

**Special token prefixes** are prepended at download time (not during cleaning or pretokenization):

| Stream | text field format |
|---|---|
| Bangla mono | `<\|lang_bn\|>{text}` |
| English mono | `<\|lang_en\|>{text}` |
| BN→EN | `<\|task_translate_bn_en\|><\|lang_bn\|>{bn}<\|lang_en\|>{en}` |
| EN→BN | `<\|task_translate_en_bn\|><\|lang_en\|>{en}<\|lang_bn\|>{bn}` |

Each NMT pair produces **two lines** — one per translation direction.

**Source-level filters applied during download:**

- **Bangla mono** (TituLLM, Wikipedia): Minimum 50 words; maximum 30% non-Bangla characters (catches misclassified documents); preserves paragraph structure.
- **English mono** (FineWeb): Minimum 50 words, maximum 100,000 words (rejects pathological outliers); NFC normalization only.
- **NMT pairs** (NLLB, BanglaNMT): Length filter (3–150 words per side); length ratio filter (0.4–2.5, accounting for Bangla morphological density); exact dedup on Bangla side via MD5 hash; both translation directions written per pair.

### 3.2 Stage 2: Unicode Normalization

**NFC (Canonical Decomposition followed by Canonical Composition)** is applied as the first normalization step on all streams. NFC is chosen over NFKC because NFKC collapses compatibility characters that can alter Bangla conjunct forms in legacy-converted text [preprocessing_steps.md].

For Bangla text, a separate **Bangla-specific Unicode normalization** pass (`bnunicodenormalizer`) is applied after download. This fixes edge cases that NFC alone cannot handle:

- Broken diacritics (e.g., `া` encoded as two separate characters)
- Invalid hosonto (hasanta) sequencing from inconsistent keyboard encodings
- Nukta normalization for standard Bangla characters
- Zero-width non-joiner (ZWNJ) and zero-width joiner (ZWJ) inconsistencies

The Bangla-specific normalizer is applied per-word to avoid `ValueError` from multi-word inputs. This normalization is applied as a post-download step to avoid the performance penalty during the streaming download phase.

### 3.3 Stage 3: Cross-Source Deduplication

Two deduplication passes are performed to remove duplicates that span data sources.

**Monolingual dedup** (TituLLM ∩ Wikipedia): Exact deduplication via SHA-256 hash on full document text. This catches documents that appear in both TituLLM's Common Crawl extraction and Wikipedia. Only the first occurrence (TituLLM, processed first) is kept.

**NMT dedup** (NLLB ∩ BanglaNMT): Exact deduplication via MD5 hash on the full text field (including special tokens). NLLB is processed first to populate the seen-hash set, so BanglaNMT only contributes pairs with Bangla sides not already present in NLLB.

Fuzzy/MinHash deduplication is deliberately **not** applied:
- TituLLM is already deduplicated by its creators
- NMT pairs are either exact duplicates or genuinely different translations
- Fuzzy matching would incorrectly drop valid stylistically similar documents

### 3.4 Stage 4: Tokenizer Training

The tokenizer is trained on a **sampled subset** of the downloaded data, not the full corpus.

**Sampling strategy:**
1. Reservoir-sample 425M words from Bangla sources (TituLLM + Wikipedia) and 75M words from English sources (FineWeb-Edu), totaling 500M words (85:15 BN:EN ratio maintained)
2. The sampler (`02_tokenizer_sampler.py`) performs uniform random sampling from cleaned JSONL files, then shuffle-merges the two language streams with a weighted random merge (probability of selecting Bangla = 85%)
3. The resulting corpus is a representative sample of the target 85:15 training composition: 1,249,199 Bangla docs + 96,205 English docs = 1,345,404 docs total (7.4 GB JSONL)

**Tokenizer configuration:**

| Parameter | Value |
|---|---|
| Type | SentencePiece Unigram |
| Vocab size | 48,000 (146 special + 47,854 learned subwords) |
| Character coverage | 0.9999 |
| Byte fallback | Yes (handles out-of-vocabulary characters) |
| Normalization rule | Identity (no internal NFC — applied externally) |
| Input sentence size | 0 (all entries loaded; no reservoir sampling) |
| Max sentence length | 65,536 chars (3,036 long docs skipped) |
| Corpus entries | 1,345,404 docs (500M words, ~3.1B chars, ~7.3 GB text) |

**Extraction method:** Each JSONL document is written as a single training entry — paragraphs are joined with spaces (not newlines) and no sentence-level splitting is applied. This is critical for memory: SentencePiece builds an internal lattice per training entry, so splitting 1.35M docs into sentences (as initially attempted) inflates the entry count to 32.5M, increasing per-entry lattice overhead by 24x and causing OOM. The doc-level approach keeps entry count at 1.35M, matching the actual document count.

**Training memory requirements:** The SentencePiece Unigram algorithm loads the entire corpus into a `vector<string>` and builds frequency tables and a subword lattice whose size is proportional to total character count. For 500M words (~3.1B characters, ~7.3 GB text), peak memory consumption during training reached approximately 32 GB RAM + 110 GB swap (of 120 GB total swap configured on a system with 32 GB physical RAM). The `train_extremely_large_corpus` flag was enabled. Training completed in approximately 3–4 hours on the full 1.35M-entry corpus.

The Unigram model is chosen over BPE because it provides:
- Probabilistic tokenization (each subword has a probability)
- Better handling of morphologically rich languages like Bangla
- Natural fallback to byte-level tokens for rare characters

**Special token budget:**

The 146 special tokens are organized into functional categories:

| Category | IDs | Count | Purpose |
|---|---|---|---|
| Control | 0–3 | 4 | pad, unk, bos, eos |
| ChatML | 4–10 | 7 | Instruction format tokens |
| Task | 11–22 | 12 | Task-specific tokens (sentiment, NER, translation, etc.) |
| Language | 23–28 | 6 | Language/variant tags (BN, EN, WBN, BNLS, MIX, CODE) |
| Sentiment | 29–33 | 5 | Sentiment label tokens |
| Reasoning | 34–37 | 4 | Chain-of-thought tokens |
| Structure | 38–45 | 8 | Document structure tokens |
| Reserved | 46–145 | 100 | Future use |

After training, the SentencePiece `.model` file is wrapped in a HuggingFace `PreTrainedTokenizerFast` for use by the training pipeline.

### 3.5 Stage 5: Pretokenization

Documents are tokenized and packed into fixed-length sequences for efficient GPU loading.

**Shard structure:**
- Separate directories per source type: `pretokenized/{bangla,english,nmt}/train/`
- Each shard is a 2D NumPy array of shape `(N, 2048)` with `dtype=uint16`
- Documents are tokenized with `<eos>` appended, then concatenated and chunked into 2048-token sequences
- No eval split — all data goes to train; leftover data after `max_steps` serves as eval

**Per-source-type shards** enable:
- Dataloader-level sampling weights (e.g., oversample NMT, undersample English)
- Easy composition adjustments without reprocessing
- Monitoring of per-source training dynamics

### 3.6 Stage 6: Verification

Before training, a verification script checks:
- Shard existence and correct shape `(N, 2048)`
- Correct dtype (`uint16`)
- Token range within `[0, 48000)`
- Absence of pathological all-zero rows
- Decode spot-checks to verify tokenization quality

---

## 4. Pipeline Execution Order

The complete preprocessing pipeline executes in the following order:

```
Download (parallel):
  1. python scripts/downloaders/01a_download_titulm_cc.py --doc-limit 15000000
  2. python scripts/downloaders/01b_download_wikipedia.py
  3. python scripts/downloaders/02a_download_english.py --word-budget 2_000_000_000
  4. python scripts/downloaders/03b_download_nllb.py          ← before BanglaNMT
  5. python scripts/downloaders/03a_download_banglanmt.py

Post-download:
  6. python scripts/pipeline/01c_bn_normalize.py              ← one-time Bangla normalization

Tokenizer training:
  7. python scripts/pipeline/02_tokenizer_sampler.py           ← sample 500M words (85:15 BN:EN)
  8. python -m src.tokenizer.train_tokenizer                  ← train SentencePiece Unigram
  9. python -m src.tokenizer.wrapper                          ← wrap .model → HF tokenizer

Cross-source dedup:
  10. python scripts/pipeline/01a_dedup_nmt.py                ← NLLB ∩ BanglaNMT
  11. python scripts/pipeline/01b_dedup_mono.py               ← TituLLM ∩ Wikipedia

Pretokenization + verification:
  12. python scripts/pipeline/03_pretokenize.py               ← pack into .npy shards
  13. python scripts/pipeline/04_verify.py                    ← sanity check shards

Training:
  14. Set max_steps in configs/default_training.yaml
  15. python src/train.py --model configs/banglagamba_12l.yaml \
        --training configs/default_training.yaml \
        --optimizer configs/muon_adamw.yaml \
        --data configs/default_data.yaml
```

---

## 5. Expected Data Composition

After all preprocessing, the training corpus is expected to contain approximately:

**Table 3: Expected corpus composition.**

| Stream | Source | Raw | After Filtering | Est. Tokens |
|---|---|---|---|---|
| Bangla monolingual | TituLLM | ~5.2B words | ~5.0B words | ~7.3B |
| Bangla monolingual | Wikipedia BN | ~51M words | ~48M words | ~71M |
| English monolingual | FineWeb-Edu | ~2B words | ~2B words | ~2.9B |
| NMT pairs (both dirs) | NLLB | ~62M pairs | ~35–45M pairs × 2 | ~1.0–1.2B |
| NMT pairs (both dirs) | BanglaNMT | ~2.75M pairs | ~2.5M pairs × 2 | ~85M |
| **Total** | | | | **~11.5–12.3B** |

The effective training composition by word count targets 85% Bangla, 15% English. If NMT data is over-represented relative to this target, composition is adjusted at the dataloader sampling weight level during training — not by re-filtering the JSONL.

---

## 6. Evaluation Strategy

The primary evaluation target is the **SentNoB** sentiment classification benchmark [sentnob], where the baseline to beat is BanglaBERT-base with a macro-F1 of 72.89 [7].

The SentNoB and BLUB test data are **excluded** from the training corpus to prevent data contamination.

---

## 7. Design Decisions and Rationale

### 7.1 Why Hybrid Mamba-3/GQA?

Pure attention models scale quadratically with sequence length, which is prohibitive for long-context Bangla text processing on consumer hardware. Mamba-3 provides linear-time sequence modeling for local pattern extraction, while GQA blocks enable global context aggregation with reduced KV cache (4 KV heads vs. 16 query heads). The 1:1 interleaving follows the empirical findings of Lieber et al. [10] and the hybrid architecture ablations in [anonymous2025hybrid].

### 7.2 Why Separate Shard Directories per Source Type?

Pretokenized data is stored in separate directories (`bangla/`, `english/`, `nmt/`) rather than a single merged directory. This enables:
- Runtime composition adjustment via dataloader sampling weights
- Per-source monitoring of training dynamics
- Easy ablation studies (e.g., "what happens without NMT data?")
- No reprocessing needed to change the training mix

### 7.3 Why 85:15 Bangla:English?

The 85:15 ratio balances:
- Sufficient Bangla exposure for language understanding (primary goal)
- Enough English exposure for cross-lingual transfer and English task performance
- Tokenizer vocab allocation proportional to actual training exposure
- Alignment with the observation that Bangla is morphologically richer (1 BN word ≈ 1.5–2 EN words), so word-count ratios underestimate actual information content

### 7.4 Why Unigram over BPE?

SentencePiece Unigram is chosen over BPE because it provides probabilistic tokenization, better handles morphologically rich languages, and naturally falls back to byte-level tokens for rare characters. The Unigram model also tends to produce more semantically meaningful subword splits for agglutinative languages like Bangla.

### 7.5 Why No MinHash on TituLLM?

TituLLM is already deduplicated by its creators using MinHash [16]. Running MinHash again wastes compute and risks discarding valid stylistically similar documents that are distinct but topically related.

---

## References

[7] A. Bhattacharjee et al., "BanglaBERT: A State-of-the-Art Language Model for Bengali," in Proceedings of NAACL-HLT, 2022.

[8] T. Nguyen et al., "CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages," arXiv preprint arXiv:2309.09400, 2024.

[10] O. Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model," arXiv preprint arXiv:2403.19887, 2024.

[13] FineWeb-Edu, Hugging Face. Available: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

[16] S. Nahin et al., "TituLLMs: Bangla Language Models with Extended Vocabulary," arXiv preprint arXiv:2502.11187, 2025.

[sentnob] SentNoB: Sentence-level Sentiment Analysis Dataset for Bengali. Available: https://github.com/csebuetnlp/SentNoB
