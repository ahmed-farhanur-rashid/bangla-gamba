---
language:
- bn
- en
library_name: transformers
tags:
- custom-architecture
- bangla
- mamba-3
- gqa
- language-model
license: cc-by-nc-sa-4.0
datasets:
- ahmed-farhanur-rashid/bn-foundational-pretrain-corpus
---

# BanglaGamba

BanglaGamba is a custom hybrid language model trained from scratch on a mixed corpus of approximately **9.62B tokens**, consisting of Bengali (\~7.37B), English (\~1.23B), and bilingual translation pairs (\~1.03B). The model combines **Mamba-3 (State Space Model)** and **Grouped Query Attention (GQA)** to provide strong performance across Bengali language understanding and generation tasks.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | ~200M |
| Context Length | 2048 |
| Architecture | Hybrid Mamba-3 + GQA (1:1 interleaved) |
| Primary Language | Bengali |
| Secondary Language | English |
| Training Tokens | ~9.62B |
| Training Dataset | `ahmed-farhanur-rashid/bn-foundational-pretrain-corpus` |
| License | CC BY-NC-SA 4.0 |

## Resources

- **GitHub Repository:** [ahmed-farhanur-rashid/bangla-gamba](https://github.com/ahmed-farhanur-rashid/bangla-gamba)
- **Training Dataset:** [datasets/ahmed-farhanur-rashid/bn-foundational-pretrain-corpus](https://huggingface.co/datasets/ahmed-farhanur-rashid/bn-foundational-pretrain-corpus)

## Requirements

This model uses a custom architecture and tokenizer implementation. Loading requires enabling `trust_remote_code=True`.

Install the required dependencies:

```bash
pip install transformers torch mamba-ssm causal-conv1d bnunicodenormalizer
```

> **Note:** The model was trained on text normalized using `bnunicodenormalizer` in data pipeline. Running without it may significantly degrade generation quality.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "ahmed-farhanur-rashid/bangla-gamba"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

prompt = "বাংলাদেশের রাজধানী"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Architecture

Unlike conventional Transformer-only language models, BanglaGamba combines state-space modeling with attention mechanisms:

- **Mamba-3 (State Space Model):** Efficient linear-time sequential modeling and long-range state recurrence without quadratic attention overhead.
- **Grouped Query Attention (GQA):** Improves inference efficiency through optimized key-value caching, enhanced with per-head QK-Norm and RoPE.
- **SwiGLU FFN:** Interleaved in every block for non-linear representation capacity.

This hybrid design aims to balance computational efficiency with strong language modeling performance.

## Related Models

BanglaGamba is part of a family of Bengali foundation language models.

| Model | Architecture | Description |
|------|--------------|-------------|
| **BanglaGamba** | Mamba-3 + GQA | Hybrid state-space and GQA architecture optimized for efficient Bengali language modeling. |
| **BanglaGSG** | GDN + SWA + GQA | Sibling hybrid architecture trained on the same corpus. |

- **BanglaGSG:** [tasmin-jahan/bangla-gsg](https://huggingface.co/tasmin-jahan/bangla-gsg)

## Limitations

- The model expects text normalized using `bnunicodenormalizer`, consistent with the preprocessing pipeline used during training.
- Loading requires execution of custom Python modules (`modeling_banglagamba.py`, `configuration_banglagamba.py`, `tokenization_banglagamba.py`) via `trust_remote_code=True`.
- While primarily trained for Bengali, English support is intended mainly for multilingual understanding and translation-related capabilities.
- As with other large language models, outputs may occasionally be inaccurate or reflect biases present in the training data.

## Citation

If you use BanglaGamba in your research, please cite the model:

```bibtex
@misc{banglagamba2026,
  title        = {BanglaGamba},
  author       = {Ahmed Farhanur Rashid},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/ahmed-farhanur-rashid/bangla-gamba}}
}
```