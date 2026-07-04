# Analysis of Normalization Failures in the Bangla Training Corpus

## 1. Overview

This document reports findings from an analysis of tokens that failed
word-level Unicode normalization during preprocessing of a Bangla text
corpus (13,549,356 base source documents from `wiki_bangla` and `titulLM`, 
plus the subsequently processed `sangraha` corpus). Normalization was performed
using a Rust reimplementation of `bnunicodenormalizer` (Bengali.AI),
validated against the reference Python implementation prior to
corpus-scale deployment (see accompanying validation report). The
purpose of this analysis is to characterize the nature of tokens that
the normalizer rejected (i.e., normalized to `None`), and to determine
whether the observed failure rate and failure composition are
consistent with expected normalizer behavior on real-world,
heterogeneously-sourced text, or whether they indicate a defect in the
normalization pipeline.

## 2. Method

Every token that the normalizer returned `None` for was logged with its
source corpus, its position in the token stream of its containing
document, and the literal token string. Document identifiers were not
retained in the failure log, which constrains the analysis to
token-level and corpus-level aggregation; document-level clustering
analysis (e.g., identifying whether failures concentrate in a small
number of severely corrupted documents versus being broadly distributed
across the corpus) was not possible with the available data and is
noted as a limitation in Section 6.

The resulting failure log (208,004 records) was analyzed along four
axes: (i) source corpus distribution, (ii) token length distribution,
(iii) character-composition of multi-character failing tokens, and (iv)
positional distribution within documents. Token classification labels
(e.g., "standalone vowel sign," "standalone nasalization") were derived
by cross-referencing each failing token's constituent Unicode codepoints
against the Unicode Bengali block's character categories.

## 3. Summary Statistics

| Metric | Value |
|---|---|
| Total source documents | 13,549,356 |
| Total failing tokens (Base Corpus) | 208,004 |
| Document-level failure rate | 1.535% |
| Unique failing token types | 235 |
| Failures from `titulLM` | 207,209 (99.62%) |
| Failures from `wiki_bangla` | 795 (0.38%) |
| Corpus size reduction post-normalization | 78.7 GB → 76.9 GB (−2.3%) |

*(Note: A separate analysis run on the `sangraha` corpus yielded an additional 428,344 failing tokens across 310 unique token types. These followed identical composition patterns to the base corpus.)*

The document-level failure rate figure (1.535%) should be interpreted
with care: it is computed as failing tokens over total documents, not
as a proportion of documents discarded. Normalization operates at
word-token granularity; a single document containing one anomalous
token is not itself discarded, only that token is. The overall corpus
size reduction (2.3% by byte count) is consistent with a
token-level, rather than document-level, filtering process, and is the
more reliable indicator of actual data loss.

## 4. Composition of Failing Tokens

### 4.1 Length distribution

| Category | Count | % of failures |
|---|---|---|
| Single-character tokens | 202,012 | 97.12% |
| Multi-character tokens | 5,992 | 2.88% |

The overwhelming majority of failures are single Unicode codepoints
that the normalizer judged invalid in isolation.

### 4.2 Character class of single-character failures

Reclassification of the 235 unique failing token types against Unicode
Bengali character categories shows failures concentrate in three
categories, none of which constitute valid standalone Bangla text under
any orthographic convention:

| Category | Approx. share of failures | Representative tokens |
|---|---|---|
| Standalone nasalization marks | ~53% | ঃ (VISARGA), ং (ANUSVARA), ঁ (CANDRABINDU) |
| Standalone vowel signs (matras) | ~35% | ে, া, ি, ী, ু, ৃ, ো |
| Standalone virama / other marks | ~5% | ্ (HALANT/VIRAMA), ় (NUKTA), ৗ (AU LENGTH MARK) |

Bangla orthography does not permit vowel signs, nasalization marks, or
the virama to occur without a preceding base consonant or vowel letter;
these are combining marks by definition, not independently valid
graphemes. A normalizer correctly implementing Bangla script rules is
expected to reject such tokens, since no well-formed rendering or
transliteration of a standalone combining mark exists. Their presence
in the source corpus, prior to normalization, indicates upstream
tokenization or text-extraction artifacts — most plausibly, a base
character was separated from its dependent mark by a line break,
whitespace-normalization step, or an encoding error during corpus
collection, leaving an orphaned mark to be tokenized as if it were a
standalone word.

### 4.3 Multi-character failures: repetition pattern

Of the 5,992 multi-character failing tokens, 5,273 (88.0%) consist of
a single Unicode character repeated `n` times with no other content
(e.g., `ঃঃঃঃঃ`, `্্্্্্্্্্`). Run lengths for this category range from
2 to 95 repetitions (median 2, mean 2.46). The remaining 719
multi-character failures (12.0%) combine two or more distinct
diacritic characters in sequence without an intervening base character
(e.g., `িং`, `েঁ`).

Repeated-character runs of this kind — particularly the longer runs
observed (up to 95 consecutive repetitions of a single combining mark)
— are not attributable to any known linguistic phenomenon in Bangla and
are consistent with corrupted or degenerate text, such as encoding
errors, OCR failure on damaged source material, or malformed
boilerplate/spam content retained during corpus collection.

## 5. Source Corpus Distribution

Failures are asymmetrically distributed across the two constituent
corpora: `titulLM` accounts for 99.62% of all failures despite not
constituting the entirety of the source corpus, while `wiki_bangla`
contributes 0.38%. This asymmetry is consistent with the general
expectation that a curated, editorially-reviewed corpus (Wikipedia)
would contain substantially less encoding noise and malformed text than
a broader, less curated web-scale corpus. The near-total absence of
failures in `wiki_bangla` (795 of over an estimated multi-million
Wikipedia-derived tokens, the large majority of them single
standalone marks rather than repetition artifacts) serves as an
informal negative control: it indicates that the normalizer does not
spuriously reject well-formed text at any appreciable rate, since the
failure rate several orders of magnitude below the noisier corpus.

Furthermore, a subsequent analysis of the `sangraha` corpus (428,344 failures) showed a near-identical composition profile to `titulLM` (approx. 56% standalone nasalization, 34% standalone vowel signs), further confirming that the normalizer consistently catches the same orthographic artifacts across diverse web-crawled datasets.

## 6. Positional Analysis and Limitations

Failing tokens' positions within their containing documents were
examined to test whether failures cluster near document boundaries
(which would suggest a truncation or document-boundary-parsing defect
in the preprocessing pipeline rather than genuinely corrupted source
text). Only 6.5% of `titulLM` failures and 0.5% of `wiki_bangla`
failures occur within the first ten token positions of their document;
the majority (64.6% and 85.8%, respectively) occur beyond position 200.
This distribution does not support a document-start truncation or
header-parsing artifact as the source of failures, and is instead
consistent with noise distributed throughout document bodies.

Two limitations constrain the strength of these conclusions. First,
document identifiers were not retained in the failure log, precluding
document-level clustering analysis; it is not possible to determine
from this data alone whether failures are broadly distributed across
many documents (consistent with sparse, scattered noise) or concentrated
in a small number of severely corrupted documents (which would suggest
a narrower but more severe upstream data-quality issue localized to a
specific source or ingestion batch). Second, the failure log records
only the rejected token and not its surrounding context, so
hypothesized causes (line-break-induced mark separation, OCR failure,
encoding corruption) are inferred from token composition and are not
directly confirmed against source context. Retaining document
identifiers and a small window of surrounding tokens in future
normalization runs would allow both limitations to be addressed at
negligible additional storage cost relative to the size of the
underlying corpus.

## 7. Conclusion

The observed normalization failures are concentrated in categories that
are, by the rules of Bangla orthography, incapable of representing
valid standalone text: orphaned combining marks (vowel signs,
nasalization marks, virama) and repeated-character degenerate
sequences. The failure rate (1.535% of documents affected; 2.3%
reduction in corpus size by byte count) is small in absolute terms and
is concentrated almost entirely in the less-curated constituent corpus,
with a negligible failure rate in the curated comparison corpus. Taken
together, this evidence supports the conclusion that the normalizer is
correctly identifying and discarding genuinely malformed tokens rather
than incorrectly rejecting well-formed Bangla text. No corrective
action to the normalization pipeline is indicated by this analysis. The
retained failure log and this analysis are preserved as a validation
artifact independent of the corpus itself, permitting this conclusion
to be revisited without requiring retention of the pre-normalization
corpus.
