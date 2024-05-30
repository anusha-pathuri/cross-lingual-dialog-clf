## Cross-Lingual Domain Classification of Task-Oriented Dialog (EN-FR)

Accurate identification of the domain of user commands is a crucial first step in digital voice assistant systems. Consider domain identification as a multi-class classification problem where the inputs are transcribed utterances. Supervised learning is limited by the availability of annotated datasets, an issue that is exacerbated for non-English languages. 

In this project, I tackle the problem of domain classification for French, a _relatively_ lower resource language than English. With access to a parallel annotated dataset in French, I set out to compare the performance gap between fully-supervised training in the target language and cross-lingual zero-shot transfer from the source language using massively pre-trained Transformer-based masked language models (PLMs). 

The results reinforce that given the recent advancements in PLMs, domain identification is a somewhat trivial task with fully-supervised fine-tuning in the target language achieving near-perfect results (~0.98 F1) and even zero-shot transfer from the source not lagging far behind (~0.95 F1).

See [Technical Report](Technical_Report.pdf) for more details on methodology and results.


### Dataset

[MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark](https://aclanthology.org/2021.eacl-main.257) (Li et al., EACL 2021) is a multi-lingual, multi-domain task-oriented dialog dataset consisting of synthetic utterances in 6 languages across 11 domains with fine-grained _intent_ labels. Different from the authors' work, this project will focus on the more coarse-grained _domain_ prediction. Dataset with domain labels is available on HuggingFace [mteb/mtop_domain](https://huggingface.co/datasets/mteb/mtop_domain).
