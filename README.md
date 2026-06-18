# Bangla-Gamba

**A ~199M parameter Mamba-3 / GQA hybrid Bangla foundation model.**

Formerly BanglaFM. A Bangla-centric language model pretrained from scratch on a multilingual corpus (Bangla + English + Banglish + code-mixed). First foundation model with explicit Banglish (romanized Bangla) pretraining.

## Architecture

| Property | Value |
|---|---|
| Architecture | Mamba-3 / GQA hybrid, 1:1 interleaved |
| Total blocks | 12 (6 Mamba-3 + 6 GQA, each with SwiGLU FFN) |
| Hidden size | 1024 |
| Total parameters | ~199M |
| Vocabulary size | 48,000 |
| Sequence length | 2048 |
| Target | Beat BanglaBERT-base (SentNoB macro-F1 72.89) |
| Hardware | RTX 4070 Super, 12GB VRAM |
| Optimizer | Hybrid Muon (2D weights) + AdamW (everything else) |

## Project Structure

```
BanglaGamba/
├── configs/
│   ├── banglagamba_12l.yaml        # Model architecture
│   ├── muon_adamw.yaml             # Optimizer config
│   ├── default_training.yaml       # Training hyperparams
│   └── default_data.yaml           # Data paths
├── src/
│   ├── train.py                    # Training entry point
│   ├── model/
│   │   ├── config.py               # BanglaGambaConfig
│   │   ├── model.py                # BanglaGambaModel
│   │   ├── attention.py            # GQA with QK-Norm
│   │   ├── ffn.py                  # SwiGLU FFN
│   │   ├── mamba.py                # Mamba-3 wrapper
│   │   ├── embeddings.py           # RMSNorm + TokenEmbedding
│   │   ├── rope.py                 # RoPE
│   │   └── optim.py                # Muon + AdamW factory
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   ├── checkpoint.py           # Checkpoint save/load
│   │   └── scheduler.py            # LR schedule
│   ├── data/
│   │   ├── dataset.py              # ShardedNpyDataset
│   │   └── collator.py             # DataLoader builder
│   └── utils/
│       ├── logging.py              # Metric logging
│       └── seed.py                 # Reproducibility
└── docs/
    ├── BanglaFM_Model_Implementation_Spec.md
    ├── BanglaFM_Complete_Guide.md
    └── BanglaFM_Q1_Data_Plan.md
```

## Quick Start

```bash
cd BanglaGamba/

# Training (requires pretokenized data in data/tokenized/)
python src/train.py \
    --model configs/banglagamba_12l.yaml \
    --training configs/default_training.yaml \
    --optimizer configs/muon_adamw.yaml \
    --data configs/default_data.yaml

# Resume from checkpoint
python src/train.py --resume [same config args]
```

## Key Features

- **Mamba-3 / GQA hybrid**: 1:1 interleaved Mamba-3 SSM and Grouped Query Attention
- **QK-Norm**: Per-head RMSNorm on Q/K projections for Muon optimizer stability
- **Z-loss**: Logit magnitude penalty preventing late-training loss spikes
- **Hybrid optimizer**: `torch.optim.Muon` for 2D weights + `torch.optim.AdamW` for rest
- **Residual init scaling**: Output projections scaled by 1/√(2·n_layers)
- **BF16 training**: With FP32 optimizer states for accumulation precision

## Dependencies

```
torch>=2.9       # for torch.optim.Muon
mamba-ssm         # for Mamba-3
pyyaml
numpy
```

## References

- Mamba-2/3: Dao & Gu, *Transformers are SSMs* (2024)
- GQA: Ainslie et al. (2023)
- SwiGLU: Shazeer (2020)
- RoPE: Su et al. (2021)
- Muon: Bernstein & Newhouse (2024)
