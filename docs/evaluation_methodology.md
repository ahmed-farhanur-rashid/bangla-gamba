# Downstream Evaluation Methodology

## 1. Overview & Evaluation Philosophy

To evaluate the representational capacity, downstream transfer efficiency, and generative capabilities of pre-trained language models for Bangla, we establish a standardized, reproducible evaluation framework (`evaluation_suit/`). The framework benchmarks models across six distinct natural language processing (NLP) tasks covering sequence classification, token classification, sentence-pair reasoning, machine translation, long-context retrieval, and abstractive summarization.

To ensure strict comparability:
1. **Identical Downstream Task Formulations**: Hyperparameters, sequence lengths, loss functions, and evaluation metrics are fixed across all evaluated architectures.
2. **Probing Paradigm for Classification**: Downstream classification tasks utilize linear probing over frozen base representations ($\nabla_{\Theta_{\text{backbone}}} \mathcal{L} = 0$). This isolates and measures the intrinsic quality of the pre-trained embedding space without confounding representation quality with task-specific backbone fine-tuning dynamics.
3. **Multi-Seed Statistical Verification**: All non-deterministic training runs are executed across three independent random seeds ($s \in \{0, 1, 2\}$) to report mean performance and standard deviation.

---

## 2. Model Taxonomies & Registry

The evaluation suite benchmarks three distinct model architectures representing different parameter scales and structural paradigms.

### Table 1: Evaluated model specifications and task eligibility.

| Model Identifier | Repository Reference | Architecture Paradigm | Backbone Parameters | Total Parameters | Context Window | Task Eligibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gamba` | `ahmed-farhanur-rashid/bangla-gamba` | Hybrid Mamba-3 + GQA (1:1 Interleaved) | 199.4M | 248.6M | 2048 | Tasks 01–06 (All) |
| `gsg` | `tasmin-jahan/bangla-gsg` | Hybrid GDN + SWA + GQA | 185.6M | 234.7M | 2048 | Tasks 01–06 (All) |
| `banglabert` | `csebuetnlp/banglabert` | Masked LM (ELECTRA Discriminator) | 110.7M | 110.7M | 512 | Tasks 01–03 (Classification only) |

- **`BanglaGamba`** (`gamba`): A 12-layer hybrid model featuring 6 Mamba-3 selective state space blocks interleaved 1:1 with 6 Grouped Query Attention (GQA) blocks ($d_{\text{model}} = 1024, n_{\text{heads}} = 16, n_{\text{kv\_heads}} = 4$). Tied embedding weights share $48,000 \times 1024$ parameters between input embeddings and output projection.
- **`BanglaGSG`** (`gsg`): A 12-layer hybrid model utilizing Gated Delta Net (GDN), Sliding Window Attention (SWA), and GQA blocks ($d_{\text{model}} = 1024$). Tied embedding weights share $48,000 \times 1024$ parameters.
- **`BanglaBERT-base`** (`banglabert`): A bidirectional 12-layer ELECTRA discriminator model ($d_{\text{model}} = 768, n_{\text{heads}} = 12$) pre-trained on Bangla web text. Evaluated exclusively on sequence and token classification tasks (Tasks 01–03).

---

## 3. Linear Probing & Training Protocol

For Tasks 01, 02, and 03, the base model backbone weights $\Theta_{\text{backbone}}$ are held frozen. A linear projection layer $W$ is attached to the extracted hidden representations $H \in \mathbb{R}^{B \times T \times d_{\text{hidden}}}$.

### 3.1 Representation Pooling Strategies

The extracted hidden representation vector $h \in \mathbb{R}^{d_{\text{hidden}}}$ passed to the linear classifier is computed based on the architecture model type:

1. **Causal Language Models (`gamba`, `gsg`)**: Last non-padding token pooling.
   $$\tau = \sum_{i=1}^T \mathbb{I}(\text{token}_i \neq \text{PAD}) - 1$$
   $$h = H_{b, \tau, :}$$
   where $\tau$ is the zero-indexed position of the final valid token in sequence $b$.

2. **Masked Language Models (`banglabert`)**: $[\text{CLS}]$ token pooling.
   $$h = H_{b, 0, :}$$

3. **Per-Token Classification (Task 02 NER)**: No spatial pooling is applied. The classification head maps each hidden state $H_{b, t, :} \in \mathbb{R}^{d_{\text{hidden}}}$ directly to tag logits $Z_{b, t, :} \in \mathbb{R}^{K}$.

### 3.2 Optimization & Model Selection

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, weight decay $= 0.01$).
- **Learning Rate**: Fixed $2.0 \times 10^{-5}$ across all classification tasks.
- **Epochs**: 5 epochs for sequence classification (Tasks 01 & 03); 10 epochs for token classification (Task 02).
- **Validation-Based Selection**: Evaluation on the validation split occurs at the termination of each epoch. The classifier parameter state $W^*$ corresponding to the peak validation metric ($F_1^{\text{val}}$) is preserved:
  $$W^* = \arg\max_{e \in \{1 \dots E\}} \text{Metric}_{\text{val}}^{(e)}$$
  Final test metrics are evaluated strictly using $W^*$.

---

## 4. Task Specifications & Benchmark Datasets

### 4.1 Task 01: Sentiment Analysis (`SentNoB`)

- **Dataset**: `SentNoB` (3-class Bangla sentiment dataset covering Positive, Negative, and Neutral classes) [Khondoker et al., 2023].
- **Data Splits**: 12,582 training / 1,573 validation / 1,573 test examples.
- **Sequence Length**: 256 tokens.
- **Primary Metric**: Macro-averaged $F_1$ score ($F_1^{\text{macro}}$).
  $$F_1^{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}, \quad C = 3$$
- **Secondary Metric**: Classification Accuracy.

### 4.2 Task 02: Named Entity Recognition (`WikiAnn-bn`)

- **Dataset**: `WikiAnn` Bangla split (`wikiann`, config `bn`) annotated with 7 BIO named entity tags: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`.
- **Data Splits**: 10,000 training / 1,000 validation / 1,000 test sentences.
- **Subword Label Alignment**: Word-level NER tags are aligned to subword tokens generated by custom SentencePiece tokenizers:
  - The first subword token of each word receives the word's ground-truth NER label.
  - Subsequent continuation subwords, special tokens ($[\text{BOS}]$, $[\text{EOS}]$), and padding positions are assigned a mask label of $-100$ and excluded from cross-entropy loss computation and metric aggregation.
- **Metric**: Entity-level $F_1$ score evaluated via strict MUC-5 precision/recall criteria using `seqeval`.

### 4.3 Task 03: Natural Language Inference & Paraphrase (`XNLI-bn` & `BanglaParaphrase`)

Evaluates sentence-pair semantic reasoning across machine-translated and native Bangla benchmarks.

1. **`XNLI-bn`**:
   - 3-class NLI dataset (`entailment`, `neutral`, `contradiction`) derived from `csebuetnlp/xnli_bn`.
   - Data Splits: 381,449 training / 2,419 validation / 4,895 test pairs.
2. **`BanglaParaphrase`**:
   - Sentence-pair paraphrase dataset derived from `csebuetnlp/BanglaParaphrase`.
   - Data Splits: 419,967 training / 23,331 validation / 23,332 test pairs.

- **Sentence-Pair Encoding**:
  - Causal LMs (`gamba`, `gsg`): Premise and hypothesis sequences are concatenated with an explicit Bangla danda delimiter:
    $$\text{Sequence} = \text{Premise} \parallel \text{" । "} \parallel \text{Hypothesis}$$
  - Masked LMs (`banglabert`): Native segment pair encoding:
    $$\text{Sequence} = [\text{CLS}] \parallel \text{Premise} \parallel [\text{SEP}] \parallel \text{Hypothesis} \parallel [\text{SEP}]$$
- **Metrics**: Accuracy and Macro $F_1$.

### 4.4 Task 04: Machine Translation (`FLORES+` ben_Beng $\leftrightarrow$ eng_Latn)

Evaluates zero-shot prompt-based translation performance on parallel sentence pairs from `openlanguagedata/flores_plus` (1,012 `devtest` sentence pairs).

- **Contamination Gating**: Prior to generation, an exact-match SHA-256 sentence hash check scans the pre-training corpus (`ahmed-farhanur-rashid/bn-foundational-pretrain-corpus`) against `devtest` sentences. Translation evaluation proceeds only if the overlap rate satisfies:
  $$r_{\text{overlap}} = \frac{|H_{\text{FLORES}} \cap H_{\text{corpus}}|}{|H_{\text{FLORES}}|} \le 0.02 \quad (2.0\%)$$
- **Prompt Formats**:
  - **BN $\to$ EN**:
    ```text
    নিম্নলিখিত বাংলা বাক্যটি ইংরেজিতে অনুবাদ করুন:

    বাংলা: {source_sentence}
    ইংরেজি:
    ```
  - **EN $\to$ BN**:
    ```text
    Translate the following English sentence to Bangla:

    English: {source_sentence}
    Bangla:
    ```
- **Decoding Configuration**: Deterministic greedy decoding ($\text{temperature} = 0.0, \text{do\_sample} = \text{False}$, $\text{max\_new\_tokens} = 128$).
- **Metrics**: SacreBLEU ($n$-gram precision with brevity penalty) and chrF (character $n$-gram $F_1$ score).

### 4.5 Task 05: Long-Context Needle-in-a-Haystack (`NIAH`)

Measures information retrieval fidelity over extended context lengths up to the model's 2048-token context boundary.

- **Dataset Construction**: Synthetically constructed haystacks formed by concatenating cleaned Bangla Wikipedia articles (`wikimedia/wikipedia`, config `20231101.bn`). A target factoid sentence ("needle", e.g., *"বাংলাদেশের রাজধানী ঢাকা।"* with target entity *"ঢাকা"*) is placed at specific relative depth positions.
- **Evaluation Grid**:
  - **Context Lengths ($L$)**: $\{256, 512, 1024, 1536, 2048\}$ tokens.
  - **Insertion Depths ($D$)**: $\{0.1, 0.3, 0.5, 0.7, 0.9\}$ (relative depth from document start to end).
  - **Sample Size**: 20 instances per grid cell ($5 \times 5 = 25$ cells, 500 total evaluation prompts).
- **Retrieval Prompt Format**:
  ```text
  {haystack_context}

  উপরের লেখা থেকে নিম্নলিখিত প্রশ্নের উত্তর দিন: লুকানো তথ্যটি কী ছিল?
  ```
- **Metric**: Retrieval accuracy based on exact string containment of the target answer entity within the generated response.

### 4.6 Task 06: Abstractive Summarization (`XL-Sum Bengali`)

Evaluates abstractive text summarization on the Bengali split of `XL-Sum` (`csebuetnlp/xlsum`, 1,012 test articles).

- **Prompt Format**:
  ```text
  নিম্নলিখিত লেখাটির সংক্ষিপ্তসার লিখুন:

  {truncated_article_text}

  সংক্ষিপ্তসার:
  ```
- **Decoding Configuration**: Source text truncated to $\text{max\_src\_len} = 1024$ tokens. Deterministic greedy decoding ($\text{max\_new\_tokens} = 256$).
- **Metrics**: ROUGE-L (Longest Common Subsequence $F_1$) and BERTScore ($F_1$, Precision, Recall using multilingual contextual embeddings).

---

## 5. Hardware & Execution Parameters

All evaluations are executed under identical hardware and precision constraints:

- **GPU Infrastructure**: NVIDIA GeForce RTX 4070 Super (12 GB VRAM).
- **Execution Precision**: Automatic Mixed Precision (`bfloat16`) for model forward passes and generation.
- **Reproducibility Controls**: Random seeds ($s \in \{0, 1, 2\}$) set deterministically across PyTorch (`torch.manual_seed`), CUDA (`torch.cuda.manual_seed_all`), NumPy (`numpy.random.seed`), and Python (`random.seed`).
- **Orchestration**: Orchestrated via [`evaluation_suit/eval/run_all.py`](file:///home/farhan/my-projects/bangla-gamba/evaluation_suit/eval/run_all.py).
