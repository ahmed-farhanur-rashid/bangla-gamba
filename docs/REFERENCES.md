# References

## Architecture Papers

### Mamba
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

### Mamba-2
```bibtex
@article{dao2024transformers,
  title={Transformers are {SSMs}: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2405.21060},
  year={2024}
}
```

### Empirical Study of Mamba-based LMs
```bibtex
@article{waleffe2024empirical,
  title={An Empirical Study of Mamba-based Language Models},
  author={Waleffe, Roger and Byeon, Wonmin and Riach, Duncan and Norick, Brandon and Korthikanti, Vijay and Dao, Tri and Gu, Albert and Hatamizadeh, Ali and Singh, Sudhakar and Narayanan, Deepak and Kulshreshtha, Garvit and Singh, Vartika and Casper, Jared and Kautz, Jan and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2406.07887},
  year={2024}
}
```

### Jamba
```bibtex
@article{lieber2024jamba,
  title={Jamba: A Hybrid Transformer-Mamba Language Model},
  author={Lieber, Opher and Lenz, Barak and Bata, Hofit and Cohen, Golan and Osin, Jhonathan and Dalmedigos, Itay and Safahi, Erez and Meirom, Shaked and Belinkov, Yonatan and Shalev-Shwartz, Shai and others},
  journal={arXiv preprint arXiv:2403.19887},
  year={2024}
}
```

### Samba
```bibtex
@article{wang2024samba,
  title={Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling},
  author={Wang, Junxiong and Yan, Jingcheng and Gu, Albert and Rush, Alexander M},
  journal={arXiv preprint arXiv:2406.07522},
  year={2024}
}
```

### Hymba
```bibtex
@article{han2024hymba,
  title={Hymba: A Hybrid-head Architecture for Small Language Models},
  author={Han, Dongcheng and Pan, Zhicheng and Li, Yandong and others},
  journal={arXiv preprint arXiv:2411.13676},
  year={2024}
}
```

### Hybrid Architecture Ablations (GQA Placement)
```bibtex
@article{anonymous2025hybrid,
  title={Hybrid Architecture Ablations: Where to Place Attention in State Space Models},
  author={Anonymous},
  journal={arXiv preprint arXiv:2510.04800},
  year={2025}
}
```

### Grouped Query Attention (GQA)
```bibtex
@inproceedings{ainslie2023gqa,
  title={{GQA}: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemel, Mikhail and Leblond, R{\'e}mi and Goswami, Vedanuj and Auli, Michael},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  doi={10.48550/arXiv.2305.13245}
}
```

### SwiGLU
```bibtex
@article{shazeer2020glu,
  title={{GLU} Variants Improve Transformer},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:2002.05202},
  year={2020}
}
```

### RMSNorm
```bibtex
@article{zhang2019rmsnorm,
  title={Root Mean Square Layer Normalization},
  author={Zhang, Biao and Sennrich, Rico},
  journal={arXiv preprint arXiv:1910.07467},
  year={2019}
}
```

### Rotary Position Embedding (RoPE)
```bibtex
@article{su2021rope,
  title={{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}
```

---

## Efficiency & Training Papers

### FlashAttention-2
```bibtex
@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```

### Mixed Precision Training (BF16)
```bibtex
@article{micikevicius2018mixed,
  title={Mixed Precision Training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and Wu, Hao},
  journal={arXiv preprint arXiv:1710.03740},
  year={2018}
}
```

### TurboQuant
```bibtex
@article{zandieh2025turboquant,
  title={{TurboQuant}: Near-Optimal Vector Quantization for {LLM} Inference},
  author={Zandieh, Amir and others},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025},
  note={ICLR 2026, Google Research}
}
```

### Chinchilla Scaling Laws
```bibtex
@article{hoffmann2022chinchilla,
  title={Training Compute-Optimal Large Language Models},
  author={Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and Casas, Diego de Las and Hendricks, Lisa Anne and Welbl, Johannes and Clark, Aidan and others},
  journal={arXiv preprint arXiv:2203.15556},
  year={2022}
}
```

---

## Data Mixing & Multilingual Papers

### CulturaX
```bibtex
@article{nguyen2024culturax,
  title={{CulturaX}: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages},
  author={Nguyen, Thuat and Van Nguyen, Chien and Lai, Viet Dac and Man, Hieu and Ngo, Nghia and Dernoncourt, Franck and others},
  journal={arXiv preprint arXiv:2309.09400},
  year={2024}
}
```

### FineWeb
```bibtex
@article{penedo2024fineweb,
  title={{FineWeb}: Decanting the Web for the Finest Text Data at Scale},
  author={Penedo, Guilherme and Kydl{\'\i}{\v{c}}ek, Hynek and Wolf, Thomas and others},
  journal={arXiv preprint arXiv:2406.17557},
  year={2024}
}
```

### Revisiting Multilingual Data Mixtures
```bibtex
@article{chang2025revisiting,
  title={Revisiting Multilingual Data Mixtures},
  author={Chang, Tyler and others},
  journal={arXiv preprint arXiv:2510.25947},
  year={2025}
}
```

### Scaling Laws for Multilingual LMs
```bibtex
@article{tanaka2024scaling,
  title={Scaling Laws for Multilingual Language Models},
  author={Tanaka, Goro and others},
  journal={arXiv preprint arXiv:2410.12883},
  year={2024}
}
```

### Optimizing Low-Resource LM Training
```bibtex
@article{fujii2024optimizing,
  title={Optimizing Low-Resource Language Model Training},
  author={Fujii, Kazuki and others},
  journal={arXiv preprint arXiv:2410.12325},
  year={2024}
}
```

### Rethinking Multilingual Continual Pretraining
```bibtex
@article{li2025rethinking,
  title={Rethinking Multilingual Continual Pretraining: The Role of Code},
  author={Li, Yihong and others},
  journal={arXiv preprint arXiv:2504.04152},
  year={2025}
}
```

### The Stack
```bibtex
@article{lozhkov2024stack,
  title={The {Stack} v2: Multilingual Source Code Datasets at Scale},
  author={Lozhkov, Anton and Ben Allal, Loubna and Li, Raymond and Kocetkov, Denis and others},
  journal={arXiv preprint arXiv:2405.15554},
  year={2024}
}
```

---

## Bangla NLP Papers

### BanglaBERT
```bibtex
@inproceedings{bhattacharjee2022banglabert,
  title={{BanglaBERT}: A State-of-the-Art Language Model for Bengali},
  author={Bhattacharjee, Abhik and Hasan, Tahmid and Ahmad, Wasi Uddin and Mubasshir, Kazi Samin and Islam, Md Saiful and Iqbal, Anindya and Rahman, M Sohel and Shahriyar, Rifat},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2022},
  doi={10.48550/arXiv.2101.00204}
}
```

### TituLLMs
```bibtex
@article{nahin2025titullm,
  title={{TituLLMs}: Bangla Language Models with Extended Vocabulary},
  author={Nahin, Shahriar Hossain and others},
  journal={arXiv preprint arXiv:2502.11187},
  year={2025}
}
```

### BongLLaMA
```bibtex
@article{zehady2024bongllama,
  title={{BongLLaMA}: A Continually Pretrained {LLaMA} for Bengali Language Understanding},
  author={Zehady, Aman and others},
  journal={arXiv preprint arXiv:2410.21200},
  year={2024}
}
```

### BanglishRev
```bibtex
@inproceedings{shamael2024banglishrev,
  title={{BanglishRev}: A Large-Scale Bangla-English and Code-Mixed E-Commerce Review Dataset},
  author={Shamael, Md Sajid and others},
  booktitle={Proceedings of the 2024 NeurIPS Datasets and Benchmarks Track},
  year={2024},
  doi={10.48550/arXiv.2411.01011}
}
```

### BnSentMix
```bibtex
@article{nasr2024bnsentmix,
  title={{BnSentMix}: A Multilingual Sentiment Analysis Dataset for Bengali Code-Mixed Texts},
  author={Nasr, Abdullah and others},
  journal={arXiv preprint arXiv:2408.08964},
  year={2024}
}
```

### BLUB Benchmark
```bibtex
@misc{khandaker2024blub,
  title={{BLUB}: Bengali Language Understanding Benchmark},
  author={Khandaker, Md Tariqul Islam and others},
  howpublished={\url{https://github.com/csebuetnlp/BLUB}},
  year={2024}
}
```

---

## Datasets

### CulturaX (bn)
```bibtex
@misc{culturax_bn,
  title={CulturaX -- Bengali (bn) subset},
  author={Nguyen, Thuat and others},
  howpublished={\url{https://huggingface.co/datasets/uonlp/CulturaX}},
  year={2024}
}
```

### FineWeb-Edu
```bibtex
@misc{fineweb_edu,
  title={FineWeb-Edu},
  author={Penedo, Guilherme and Kydl{\'\i}{\v{c}}ek, Hynek and Wolf, Thomas and others},
  howpublished={\url{https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu}},
  year={2024}
}
```

### BanglishRev
```bibtex
@misc{banglishrev_dataset,
  title={Bangla-English and Code-Mixed E-Commerce Review Dataset},
  author={Shamael, Md Sajid and others},
  howpublished={\url{https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset}},
  year={2024}
}
```

### BanglaBERT Corpus (BUET)
```bibtex
@misc{banglabert_corpus,
  title={BanglaBERT Dataset},
  author={Bhattacharjee, Abhik and Hasan, Tahmid and Ahmad, Wasi Uddin and others},
  howpublished={\url{https://huggingface.co/datasets/csebuetnlp/banglabert_dataset}},
  year={2022}
}
```

### OSCAR 23.01 (bn)
```bibtex
@misc{oscar2301_bn,
  title={{OSCAR} 2301 -- Bengali (bn) subset},
  author={Abadji, Julien and Suarez, Pedro Ortiz and Romary, Laurent and Sagot, Beno{\^\i}t},
  howpublished={\url{https://huggingface.co/datasets/oscar-corpus/OSCAR-2301}},
  year={2023}
}
```

### CC-100 (bn)
```bibtex
@misc{cc100_bn,
  title={{CC-100} -- Bengali (bn) subset},
  author={Wenzek, Guillaume and Lachaux, Marie-Anne and Conneau, Alexis and Chaudhary, Vishrav and Guzm{\'a}n, Francisco and Joulin, Armand and Grave, {\'E}douard},
  howpublished={\url{https://huggingface.co/datasets/statmt/cc100}},
  year={2020}
}
```

### Wikipedia (bn)
```bibtex
@misc{wikimedia_bn,
  title={Wikimedia Downloads -- Bengali Wikipedia},
  author={Wikimedia Foundation},
  howpublished={\url{https://huggingface.co/datasets/wikimedia/wikipedia}},
  year={2024}
}
```

### Sangraha (bn)
```bibtex
@misc{sangraha_bn,
  title={Sangraha -- Bengali (bn) subset},
  author={AI4Bharat},
  howpublished={\url{https://huggingface.co/datasets/ai4bharat/sangraha}},
  year={2024}
}
```

### BanglaTLit
```bibtex
@misc{banglatlit,
  title={{BanglaTLit}: Bengali Transliteration Dataset},
  author={SBNLTK},
  howpublished={\url{https://huggingface.co/datasets/sbnltk/BanglaTLit}},
  year={2024}
}
```

### SKNahin Bengali Transliteration
```bibtex
@misc{sknahin_translit,
  title={Bengali Transliteration Data},
  author={Nahin, Shahriar Hossain},
  howpublished={\url{https://huggingface.co/datasets/SKNahin/bengali-transliteration-data}},
  year={2024}
}
```

### NCTB Textbooks (TigerLLM)
```bibtex
@misc{nctb_textbooks,
  title={{NCTB} Textbooks -- Bangladeshi National Curriculum},
  author={Raihan, Md},
  howpublished={\url{https://github.com/mraihan-gmu/TigerLLM}},
  year={2024}
}
```

### Dakshina (bn)
```bibtex
@misc{dakshina_bn,
  title={Dakshina -- Bengali (bn)},
  author={Roark, Brian and Wolf-Sonkin, Lawrence and Kirov, Christo and Sproat, Richard},
  howpublished={\url{https://github.com/google-research-datasets/dakshina}},
  year={2020}
}
```

### Ayon128/Banglish-English
```bibtex
@misc{ayon128_banglish,
  title={Banglish-English Dataset},
  author={Ayon128},
  howpublished={\url{https://huggingface.co/datasets/Ayon128/Banglish-English}},
  year={2024}
}
```

---

## Evaluation Benchmarks

### SentNoB
```bibtex
@misc{sentnob,
  title={{SentNoB}: Sentence-level Sentiment Analysis Dataset for Bengali},
  author={CSE BUET NLP Group},
  howpublished={\url{https://github.com/csebuetnlp/SentNoB}},
  year={2023}
}
```

### BanglaGLUE
```bibtex
@misc{banglglue,
  title={{BanglaGLUE}: A Suite of Bengali Language Understanding Tasks},
  author={CSE BUET NLP Group},
  howpublished={\url{https://github.com/csebuetnlp/banglabert}},
  year={2022}
}
```

### MultiCoNER v2 (bn)
```bibtex
@inproceedings{multiconer2023,
  title={Overview of {MultiCoNER} v2: Multilingual Complex Named Entity Recognition},
  author={Shao, Tong and others},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}
```

---

## Tools & Libraries

### SentencePiece
```bibtex
@article{kudo2018sentencepiece,
  title={Sentence{P}iece: A simple and language independent subword tokenizer and detokenizer for neural text processing},
  author={Kudo, Taku and Richardson, John},
  journal={arXiv preprint arXiv:1808.06226},
  year={2018}
}
```

### bnunicodenormalizer
```bibtex
@misc{bnunicodenormalizer,
  title={bnunicodenormalizer: Bengali Unicode Normalizer},
  author={Sagor, Md Hasan},
  howpublished={\url{https://pypi.org/project/bnunicodenormalizer/}},
  year={2023}
}
```

### Aksharamukha
```bibtex
@misc{aksharamukha,
  title={Aksharamukha: Script Transliteration Library},
  author={Vinodh Rajan},
  howpublished={\url{https://github.com/virtualvinodh/aksharamukha}},
  year={2023}
}
```

### trafilatura
```bibtex
@inproceedings{barbaresi2021trafilatura,
  title={Trafilatura: A Web Scraping Library for Text Discovery and Transformation},
  author={Barbaresi, Adrien},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations},
  year={2021}
}
```
