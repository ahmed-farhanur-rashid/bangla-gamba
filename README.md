# Bangla-Gamba

**A ~200M parameter Mamba-3 / GQA hybrid Bangla mini-llm.**

Formerly BanglaFM. A Bangla-centric language model pretrained from scratch on a bilingual corpus (Bangla + English + dual-translation pairs).

The naming follows the convention of hybrid architectures like Jamba (JQA + Mamba) and Samba (SWA + Mamba).

Sibling Model: [Bangla-GSG](https://github.com/tasmin-jahan/bangla-gsg)

## Architecture

| Property | Value |
|---|---|
| Architecture | Mamba-3 / GQA hybrid, 1:1 interleaved |
| Total blocks | 12 (6 Mamba-3 + 6 GQA, each with SwiGLU FFN) |
| Hidden size | 1024 |
| Total parameters | ~200M (199.4M) |
| Vocabulary size | 48,000 |
| Sequence length | 2048 |
| Hardware | RTX 4070 Super, 12GB VRAM |
| Optimizer | Hybrid Muon (2D weights) + AdamW (everything else) |

## Project Structure

```
BanglaGamba/
├── configs/
│   ├── banglagamba_12l.yaml        # Model architecture config
│   ├── muon_adamw.yaml             # Hybrid optimizer config
│   ├── default_training.yaml       # Pretraining hyperparameters
│   └── default_data.yaml           # Dataset paths & shard configs
├── notebooks/                      # Data exploration & evaluation notebooks
│   ├── 00_data_exploration.ipynb   # Figure-heavy EDA & norm failure audit
│   ├── 01_training_metrics.ipynb   # Loss curves & learning rate plots
│   └── 02_perplexity_loss_eval.ipynb # Validation perplexity & loss evaluation
├── src/
│   ├── train.py                    # Pretraining entry point
│   ├── tokenizer/                  # Tokenizer training & normalization utilities
│   ├── model/                      # Model core components
│   │   ├── config.py               # BanglaGambaConfig
│   │   ├── model.py                # BanglaGambaModel
│   │   ├── attention.py            # GQA with QK-Norm
│   │   ├── ffn.py                  # SwiGLU FFN
│   │   ├── mamba.py                # Mamba-3 SSM block wrapper
│   │   ├── embeddings.py           # RMSNorm + TokenEmbedding
│   │   ├── rope.py                 # Rotary Position Embeddings
│   │   └── optim.py                # Muon + AdamW optimizer factory
│   ├── hf_integration/             # Hugging Face integration code & auto_map
│   │   ├── configuration_banglagamba.py # HF BanglaGambaConfig
│   │   ├── modeling_banglagamba.py     # HF BanglaGambaForCausalLM
│   │   └── tokenization_banglagamba.py # HF BanglaGambaTokenizer
│   ├── training/
│   │   ├── trainer.py              # Main training loop
│   │   ├── checkpoint.py           # Checkpoint save/load logic
│   │   └── scheduler.py            # Warmup + Cosine LR scheduler
│   ├── data/
│   │   ├── dataset.py              # ShardedNpyDataset memory-mapped loader
│   │   └── collator.py             # PyTorch DataLoader builder
│   └── utils/
│       ├── logging.py              # TensorBoard / CSV metric logging
│       └── seed.py                 # Reproducibility seed setup
├── utils/                          # CLI utilities & Hugging Face upload tools
│   ├── eda_helpers.py              # Reusable EDA statistical auditing functions
│   ├── count_tokens.py             # Dataset shard token counter
│   ├── prepare_hf_upload.py        # Hugging Face staging converter
│   └── convert_config_to_json.py   # Config YAML -> JSON converter
├── saved/                          # Pretrained artifacts, logs, and reports
│   ├── model/                      # Checkpoints & model weights
│   ├── reports/                    # YAML token counts & norm failure reports
│   └── logs/                       # Training logs & norm failure JSONL logs
├── pretrain-corpus-pipeline/       # Dataset download & preprocessing pipeline
├── scripts/                        # Data packing & pretokenization scripts
└── tests/                          # Evaluation and verification test suite
```

## Quick Start

```bash
cd BanglaGamba/

# Pretraining with default configs
python src/train.py

# Pretraining with custom configs
python src/train.py \
    --model configs/custom_model_config.yaml \
    --training configs/custom_training_config.yaml \
    --optimizer configs/custom_optimizer_config.yaml \
    --data configs/custom_data_config.yaml

# Resume pretraining from latest checkpoint
python src/train.py --resume

# Resume with custom configs
python src/train.py \
    --model configs/custom_model_config.yaml \
    --training configs/custom_training_config.yaml \
    --optimizer configs/custom_optimizer_config.yaml \
    --data configs/custom_data_config.yaml \
    --resume
```

## Key Features

- **Mamba-3 / GQA Hybrid Architecture**: 1:1 interleaved Mamba-3 State-Space Model blocks and Grouped Query Attention layers with SwiGLU FFNs.
- **Hugging Face `auto_map` Integration**: Native integration with `AutoConfig`, `AutoModelForCausalLM`, and `AutoTokenizer` supporting `trust_remote_code=True`.
- **Integrated Unicode Normalization**: Custom `BanglaGambaTokenizer` automatically applies `bnunicodenormalizer` during pre-tokenization and strips Metaspace `▁` symbols during decoding.
- **QK-Norm**: Per-head RMSNorm on Q/K projections for Muon optimizer stability.
- **Z-loss**: Logit magnitude penalty preventing late-training loss spikes.
- **Hybrid Optimizer**: `torch.optim.Muon` for 2D matrix weights + `torch.optim.AdamW` for embeddings, layer norms, and biases.
- **Residual Initialization Scaling**: Output projection weights scaled by $1 / \sqrt{2 \cdot N_{\text{layers}}}$.
- **BF16 Training**: Mixed precision pretraining with FP32 optimizer accumulation states.
- **Automated Data Auditing & EDA**: Comprehensive token counting, mmap shape verification, and figure-heavy EDA notebooks for pretraining data analysis.

## Dependencies

```
torch>=2.1.0
transformers>=4.40.0
mamba-ssm>=2.2.4
causal-conv1d>=1.4.0
bnunicodenormalizer
pyyaml
numpy
pandas
matplotlib
seaborn
```

## References

- Mamba-2/3: Dao & Gu, *Transformers are SSMs* (2024)
- GQA: Ainslie et al. (2023)
- SwiGLU: Shazeer (2020)
- RoPE: Su et al. (2021)
- Muon: Bernstein & Newhouse (2024)
