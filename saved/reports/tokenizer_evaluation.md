# BanglaGamba Tokenizer Evaluation

**Date:** 2026-07-05 18:55:31


## Tokenizer Summary

| Tokenizer | Vocab Size | Type |
|---|---|---|
| BanglaGamba | 48,000 | Unigram (SP) |
| mBART-50 | 250,054 | Auto |
| NLLB-200 | 256,204 | Auto |
| BanglaBERT | 101,975 | Auto |
| GPT-2 | 50,257 | Auto |

## Curated Sentence Tests

### Fertility (tokens/word, lower = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| bangla_formal | 1.953 | 1.884 | 13.442 | 1.581 | 1.465 |
| bangla_news | 2.152 | 1.333 | 14.636 | 2.061 | 1.848 |
| english | 1.102 | 1.245 | 1.184 | 1.367 | 1.347 |
| banglish | 1.171 | 2.400 | 2.143 | 1.714 | 1.657 |
| code_mixed | 1.440 | 1.400 | 6.340 | 1.340 | 1.380 |
| python_code | 3.138 | 3.414 | 2.586 | 2.897 | 3.069 |
| bangla_edge_cases | 2.615 | 2.462 | 9.962 | 2.808 | 2.846 |

### Compression (chars/token, higher = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| bangla_formal | 3.42 | 3.54 | 0.50 | 4.22 | 4.56 |
| bangla_news | 3.30 | 5.32 | 0.48 | 3.44 | 3.84 |
| english | 5.61 | 4.97 | 5.22 | 4.52 | 4.59 |
| banglish | 4.76 | 2.32 | 2.60 | 3.25 | 3.36 |
| code_mixed | 3.50 | 3.60 | 0.79 | 3.76 | 3.65 |
| python_code | 2.34 | 2.15 | 2.84 | 2.54 | 2.39 |
| bangla_edge_cases | 2.63 | 2.80 | 0.69 | 2.45 | 2.42 |

### UNK Rate % (lower = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| bangla_formal | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bangla_news | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| english | 83.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| banglish | 82.9268 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| code_mixed | 20.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| python_code | 34.0659 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bangla_edge_cases | 26.4706 | 0.0000 | 0.0000 | 2.7397 | 0.0000 |

### Round-Trip Failure % (lower = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| bangla_formal | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bangla_news | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| english | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| banglish | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| code_mixed | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| python_code | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bangla_edge_cases | 100.00 | 0.00 | 0.00 | 40.00 | 20.00 |

### Hyper-Fragmentation % (words > 3 tokens)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| bangla_formal | 6.98 | 16.28 | 100.00 | 2.33 | 0.00 |
| bangla_news | 12.12 | 6.06 | 100.00 | 9.09 | 9.09 |
| english | 0.00 | 0.00 | 0.00 | 2.04 | 2.04 |
| banglish | 0.00 | 5.71 | 5.71 | 2.86 | 2.86 |
| code_mixed | 4.00 | 4.00 | 60.00 | 0.00 | 0.00 |
| python_code | 34.48 | 37.93 | 27.59 | 27.59 | 31.03 |
| bangla_edge_cases | 26.92 | 23.08 | 69.23 | 38.46 | 34.62 |

## Corpus Tests (sampled from cleaned data)

### Fertility (tokens/word, lower = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| corpus_bangla | 1.999 | 1.380 | 12.784 | 2.114 | 2.080 |
| corpus_english | 1.206 | 1.507 | 1.342 | 1.471 | 1.465 |

### Compression (chars/token, higher = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| corpus_bangla | 3.22 | 4.67 | 0.50 | 3.05 | 3.10 |
| corpus_english | 5.11 | 4.09 | 4.59 | 4.19 | 4.21 |

### UNK Rate % (lower = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| corpus_bangla | 2.3294 | 0.0000 | 0.0000 | 0.4973 | 0.0003 |
| corpus_english | 81.2349 | 0.0000 | 0.0000 | 0.9303 | 0.0012 |

### Round-Trip Failure % (lower = better)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| corpus_bangla | 100.00 | 0.00 | 0.00 | 100.00 | 100.00 |
| corpus_english | 100.00 | 0.00 | 0.00 | 67.20 | 9.98 |

### Hyper-Fragmentation % (words > 3 tokens)

| Category | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| corpus_bangla | 11.33 | 2.28 | 96.22 | 13.32 | 14.19 |
| corpus_english | 0.94 | 5.17 | 4.30 | 3.59 | 3.33 |

## Detailed Metrics

### bangla_formal

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 1.953 | 1.884 | 13.442 | 1.581 | 1.465 |
| compression | 3.42 | 3.54 | 0.5 | 4.22 | 4.56 |
| unk_rate_pct | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hyper_frag_pct | 6.98 | 16.28 | 100.0 | 2.33 | 0.0 |
| avg_tokens_per_doc | 16.8 | 16.2 | 115.6 | 13.6 | 12.6 |
| docs_per_sec | 6990.5 | 3219.5 | 5228.5 | 7097.0 | 6066.4 |

### bangla_news

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 2.152 | 1.333 | 14.636 | 2.061 | 1.848 |
| compression | 3.3 | 5.32 | 0.48 | 3.44 | 3.84 |
| unk_rate_pct | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hyper_frag_pct | 12.12 | 6.06 | 100.0 | 9.09 | 9.09 |
| avg_tokens_per_doc | 14.2 | 8.8 | 96.6 | 13.6 | 12.2 |
| docs_per_sec | 8943.1 | 7234.1 | 7169.8 | 9316.5 | 8651.6 |

### english

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 1.102 | 1.245 | 1.184 | 1.367 | 1.347 |
| compression | 5.61 | 4.97 | 5.22 | 4.52 | 4.59 |
| unk_rate_pct | 83.3333 | 0.0 | 0.0 | 0.0 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hyper_frag_pct | 0.0 | 0.0 | 0.0 | 2.04 | 2.04 |
| avg_tokens_per_doc | 10.8 | 12.2 | 11.6 | 13.4 | 13.2 |
| docs_per_sec | 8259.8 | 6719.5 | 8837.6 | 7715.8 | 6456.7 |

### banglish

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 1.171 | 2.4 | 2.143 | 1.714 | 1.657 |
| compression | 4.76 | 2.32 | 2.6 | 3.25 | 3.36 |
| unk_rate_pct | 82.9268 | 0.0 | 0.0 | 0.0 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hyper_frag_pct | 0.0 | 5.71 | 5.71 | 2.86 | 2.86 |
| avg_tokens_per_doc | 8.2 | 16.8 | 15.0 | 12.0 | 11.6 |
| docs_per_sec | 11618.6 | 9506.6 | 11855.0 | 9887.6 | 9948.5 |

### code_mixed

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 1.44 | 1.4 | 6.34 | 1.34 | 1.38 |
| compression | 3.5 | 3.6 | 0.79 | 3.76 | 3.65 |
| unk_rate_pct | 20.8333 | 0.0 | 0.0 | 0.0 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hyper_frag_pct | 4.0 | 4.0 | 60.0 | 0.0 | 0.0 |
| avg_tokens_per_doc | 14.4 | 14.0 | 63.4 | 13.4 | 13.8 |
| docs_per_sec | 7511.3 | 6824.4 | 7044.5 | 7707.3 | 7309.7 |

### python_code

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 3.138 | 3.414 | 2.586 | 2.897 | 3.069 |
| compression | 2.34 | 2.15 | 2.84 | 2.54 | 2.39 |
| unk_rate_pct | 34.0659 | 0.0 | 0.0 | 0.0 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| hyper_frag_pct | 34.48 | 37.93 | 27.59 | 27.59 | 31.03 |
| avg_tokens_per_doc | 18.2 | 19.8 | 15.0 | 16.8 | 17.8 |
| docs_per_sec | 10771.2 | 10381.9 | 12235.4 | 10928.4 | 11250.8 |

### bangla_edge_cases

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 2.615 | 2.462 | 9.962 | 2.808 | 2.846 |
| compression | 2.63 | 2.8 | 0.69 | 2.45 | 2.42 |
| unk_rate_pct | 26.4706 | 0.0 | 0.0 | 2.7397 | 0.0 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 40.0 | 20.0 |
| hyper_frag_pct | 26.92 | 23.08 | 69.23 | 38.46 | 34.62 |
| avg_tokens_per_doc | 13.6 | 12.8 | 51.8 | 14.6 | 14.8 |
| docs_per_sec | 10749.1 | 9549.9 | 9673.2 | 10678.0 | 10136.1 |

### corpus_bangla

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 1.999 | 1.38 | 12.784 | 2.114 | 2.08 |
| compression | 3.22 | 4.67 | 0.5 | 3.05 | 3.1 |
| unk_rate_pct | 2.3294 | 0.0 | 0.0 | 0.4973 | 0.0003 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 |
| hyper_frag_pct | 11.33 | 2.28 | 96.22 | 13.32 | 14.19 |
| avg_tokens_per_doc | 659.1 | 455.1 | 4215.3 | 697.1 | 685.9 |
| docs_per_sec | 227.3 | 265.1 | 177.6 | 271.1 | 256.3 |

### corpus_english

| Metric | BanglaBERT | BanglaGamba | GPT-2 | NLLB-200 | mBART-50 |
|---|---|---|---|---|---|
| fertility | 1.206 | 1.507 | 1.342 | 1.471 | 1.465 |
| compression | 5.11 | 4.09 | 4.59 | 4.19 | 4.21 |
| unk_rate_pct | 81.2349 | 0.0 | 0.0 | 0.9303 | 0.0012 |
| round_trip_fail_pct | 100.0 | 0.0 | 0.0 | 67.2 | 9.98 |
| hyper_frag_pct | 0.94 | 5.17 | 4.3 | 3.59 | 3.33 |
| avg_tokens_per_doc | 903.7 | 1129.0 | 1005.4 | 1101.7 | 1097.6 |
| docs_per_sec | 124.6 | 138.9 | 138.3 | 132.6 | 122.6 |
