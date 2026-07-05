# **Developing a Large Language Model for Bangla: Pretraining and Fine-Tuning for NLP Applications**

Ahmed Farhanur Rashid	(0242310005101839)

### **Abstract**

The development of Large Language Models (LLMs) for the Bangla language has been largely restricted to standard Encoder only Transformer architectures. Transformers suffer from a O(n2) computational bottleneck, which causes issues for long context reasoning such as memory explosion and lost in the middle phenomenon. This paper introduces BanglaGamba, a Bangla-English-Banglish corpus pretrained foundational model to apply State Space Models (SSMs) to overcome this limitation. The proposed model is a 199M-parameter hybrid architecture that uses Mamba-3 layers with Grouped-Query Attention (GQA) in a 1:1 ratio across 12 layers, in an interleaved pattern. This hybrid design theoretically achieves both the linear-time inference of Mamba and the context retrieval capabilities of attention mechanisms. To make training feasible on consumer-grade hardware (12 GB VRAM), BanglaGamba uses mixed-precision BF16 and FlashAttention-2 during pretraining. The intent is to set a new floor for general purpose language modeling in terms of efficiency by moving towards hybrid architectures without needing massive compute.

### **Introduction**

Introduction of attention (attn) mechanisms sped up LLM development significantly. However, for resource scarce languages such as Bangla, the development has remained relatively stagnant due to lack of diverse and big datasets as well as lack of compute. The lack of compute also indirectly hurts the consumer side of the whole ordeal, inference. Prior initiatives, such as BanglaGPT, BanglaLlama, and TigerLLM, have successfully adapted standard architectures to Bangla. Yet, these models remain fundamentally held-back by the quadratic time and memory complexity of attention mechanisms leading to long inference time and high memory usage. To address these bottlenecks, BanglaGamba (199M) was made, a general purpose model that utilizes the best of both worlds, fast inference and long context reasoning from Mamba and Attn. Primary leverage here is the recent breakthroughs in State Space Models (SSMs) to enable linear-time autoregressive inference through MIMO (multiple in-multiple out) or SISO (single in-single out). A strict 1:1 interleaved architecture of Mamba-3 layers and Grouped-Query Attention (GQA) were implemented across 12 layers. Developing a foundational model on a computational budget (12 GB VRAM), requires ruthless architectural and training optimizations. Structural choices underlying BanglaGamba remain relatively standard, including the integration of SwiGLU activations, RMSNorm, and Rotary Position Embeddings (RoPE). FlashAttention-2 and BF16 remain mandatory in order to efficiently train the model from scratch. 

### **Prior Work**

| Ref. | Authors (Year) / Source | Methodology | Key Findings | Limitations |
| :---- | :---- | :---- | :---- | :---- |
| \[1\] | Bhattacharjee et al. (2022) / NAACL 2022 | Pretrained BERT-based models (BanglaBERT/BanglishBERT) on 27.5 GB Bangla corpus using the ELECTRA objective. | Outperforms multilingual models (mBERT, XLM-R) on downstream NLU tasks while being highly sample and compute-efficient. | As an encoder-only architecture, it is primarily designed for NLU and lacks autoregressive generative capabilities. |
| \[2\] | Salim et al. (2023) / ICICT4SD | Trained a language-specific GPT-2 model (BanglaGPT) on a 26.24 GB corpus (BanglaCLM) with BPE tokenization. | Achieves a perplexity of 2.86, outperforming LSTM-based sequence-to-sequence models and mGPT for Bangla text generation. | Restricted to older GPT-2 architecture; explicitly limited to a fixed context size of only 128 tokens, truncating longer documents. |
| \[3\] | Zehady et al. (2024) / arXiv | Continually pretrained LLaMA variants on the CulturaX Bangla subset and instruction-tuned on translated Alpaca/Orca datasets. | Outperforms Meta-LLaMA baselines on reasoning, Open QA, and literature tasks due to Bangla-focused tuning. | Pretrained on a modest corpus size strictly due to "computational and resource constraints" associated with the LLaMA architecture. |
| \[4\] | Nahin et al. (2025) / arXiv | Adapted Llama-3.2 (1B, 3B) via continual pretraining (\~37B tokens) and developed 5 benchmarking datasets. | The extended tokenizer significantly improves reasoning and physical commonsense capabilities compared to base models. | Explicitly reports that "performance on long contexts remains suboptimal" and cites heavy computational constraints during training/inference. |
| \[5\] | Raihan & Zampieri (2025) / arXiv | Continually pretrained LLaMA/Gemma models on a 10M token Bangla-TextBook corpus and 100K distilled Bangla instructions. | Emphasizing data quality over quantity, TigerLLM outperforms existing open-source Bangla LLMs across six standard benchmarks. | Constrained by a small pretraining corpus; notes that scaling to larger architectures is severely limited by computational constraints. |

### **Thematic Analysis**

**Low Resource LLMs:** Several prior efforts have tackled Bangla NLP with purely Transformer-based architectures. BanglaBERT \[1\] established the BLUB benchmark using an ELECTRA encoder, but lacks generative capabilities. BanglaGPT \[2\] introduced a small GPT-2 decoder limited to a 128-token context, while BanglaLlama \[3\], TituLLMs \[4\], and TigerLLM \[5\] applied continual pretraining on LLaMA variants but suffered from severe computational constraints. All these models rely on standard Transformer backbones burdened by quadratic attention complexity. BanglaGamba diverges from this lineage by adopting a hybrid Mamba-attn architecture trained from scratch on Bangla-Code mixed corpus, making it the first Bangla foundation model to leverage state space models to overcome these bottlenecks.

**State Space Models:** The introduction of Mamba \[6\] demonstrated that selective state space models (SSMs) can match Transformer-level quality while enabling linear-time autoregressive inference by avoiding the quadratic bottleneck. Dao and Gu \[7\] formalized a duality between SSMs and attention, unifying the two through structured state space duality (SSD) and deriving Mamba-2. Lahoti et al. \[8\] further refined the architecture with improved gating mechanisms and hardware-efficient kernels in Mamba-3. 

**Hybrid Architectures:** While pure SSMs are highly efficient, combining them with attention yields optimal results. Jamba \[9\] was the first to demonstrate that interleaving SSM and attention layers in a 1:1 ratio can combine the linear-decoding efficiency of Mamba with the strong in-context retrieval capacity of attention. Bae et al. \[10\] systematically analyzed design choices for hybrid models, confirming that a balanced interleaving pattern where ideally, attn block will be placed last, achieves the best perplexity-to-throughput trade-off. This hybrid paradigm directly motivates the BanglaGamba architecture, which adopts this proven 1:1 interleaved pattern (Mamba-3 followed by GQA attention) across 12 layers.

**Architectural Optimizations:** Grouped-Query Attention (GQA) \[11\] reduces the KV-cache footprint by sharing key/value heads, enabling larger batch sizes on memory-constrained GPUs. The SwiGLU activation \[12\] provides improved gradient flow and higher representation quality at a minimal parameter overhead. Root Mean Square Layer Normalization (RMSNorm) \[13\] reduces computational overhead by removing the mean-centering step while maintaining training stability. Additionally, Rotary Position Embeddings (RoPE) \[14\] provide a principled way to encode relative positions, enabling superior length generalization compared to sinusoidal embeddings.

**Training & Inference Optimization:** Training a 199M-parameter model from scratch on consumer hardware (12 GB VRAM) requires aggressive optimization beyond base architecture. Mixed precision training \[15\] with BF16 halves memory usage and doubles throughput. FlashAttention-2 \[16\] tiles the attention computation across SRAM, eliminating memory bandwidth bottlenecks and making GQA highly feasible for extended sequence lengths. TurboQuant \[17\] demonstrates that by shrinking KV cache to 3 bits, memory usage can drop 6 times while inference speed increases up to 8 times. Finally, Hoffmann et al. \[18\] established that optimal model quality follows compute-based scaling laws, in this case, the Chinchilla Optimal which suggests a 1:20 param:training\_token ratio. BanglaGamba's training corpus is carefully curated to match twitch the Chinchilla scale optimal at 1:40 ratio. 

### **Conclusion**

This paper examined the current limitations of Bangla language models, associated with inference speed and memory when using conventional Transformer-based architecture in long-context settings. To overcome this, we introduced BanglaGamba, a hybrid foundation model. Using Mamba-3 state space models and GQA in a one-to-one ratio, we brought together linear time complexity of state space models and the long context precise token retrieval.

### **References**

\[1\] A. Bhattacharjee, T. Hasan, W. Ahmad, K. S. Mubasshir, M. S. Islam, A. Iqbal, M. S. Rahman, and R. Shahriyar, "BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla," in *Findings Assoc. Comput. Linguist.: NAACL 2022*, Seattle, WA, USA, Jul. 2022, pp. 1318–1327.

\[2\] M. S. Salim, H. Murad, D. Das, and F. Ahmed, "BanglaGPT: A generative pretrained transformer-based model for Bangla language," in *Proc. Int. Conf. Inf. Commun. Technol. Sustain. Dev. (ICICT4SD)*, Dhaka, Bangladesh, 2023\.

\[3\] A. K. Zehady, S. R. Dipta, and N. I. S. A. Mamun, "BanglaLlama: LLaMA for Bangla language," *arXiv*:2410.21200, Oct. 2024\.

\[4\] S. K. Nahin et al., "TituLLMs: A family of Bangla LLMs with comprehensive benchmarking," *arXiv*:2502.11187, Feb. 2025\.

\[5\] N. Raihan and M. Zampieri, "TigerLLM: A family of Bangla large language models," *arXiv*:2503.10995, Mar. 2025\.  
\[6\] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv*:2312.00752, Dec. 2023\.

\[7\] T. Dao and A. Gu, "Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality," *arXiv*:2405.21060, May 2024\.

\[8\] A. Lahoti et al., "Mamba-3: Improved sequence modeling using state space principles," *arXiv*:2603.15569, Mar. 2026\.

\[9\] O. Lieber et al., "Jamba: A hybrid transformer-Mamba language model," *arXiv*:2403.19887, Mar. 2024\.

\[10\] S. Bae et al., "Hybrid architectures for language models: Systematic analysis and design insights," *arXiv*:2510.04800, Oct. 2024\.

\[11\] J. Ainslie et al., "GQA: Training generalized multi-query transformer models from multi-head checkpoints," *arXiv*:2305.13245, May 2023\.

\[12\] N. Shazeer, "GLU variants improve transformer," *arXiv*:2002.05202, Feb. 2020\.

\[13\] B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Adv. Neural Inf. Process. Syst.*, vol. 32, Vancouver, Canada, Dec. 2019, pp. 12360–12371.

\[14\] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, "RoFormer: Enhanced transformer with rotary position embedding," *Neurocomputing*, vol. 568, p. 127063, Feb. 2024\.

\[15\] P. Micikevicius et al., "Mixed precision training," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vancouver, Canada, 2018\.

\[16\] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," *arXiv*:2307.08691, Jul. 2023\.

\[17\] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online vector quantization with near-optimal distortion rate," in *Adv. Neural Inf. Process. Syst.*, vol. 37, Vancouver, Canada, Dec. 2024, pp. 140589–140631.

\[18\] J. Hoffmann et al., "Training compute-optimal large language models," *arXiv*:2203.15556, Mar. 2022\.

