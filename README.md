# CDA4GEC
Implementation of ACL 2024 findings ["Improving Grammatical Error Correction via Contextual Data Augmentation"](https://arxiv.org/abs/2406.17456)
<div align="center">
    <image src='./pic/main.png' width="60%">
</div>

# Requirements
The part of the model is implemented using the fairseq framework and the required environment is as follows:
- python >= 3.8
- pytorch >= 1.10.0
- spacy
- errant
- vllm
- fairseq
```
cd fairseq
pip install --editable ./
```

# Model Weights
We release the model weights of each training stage.
Our model is trained based on the Fairseq framework, details of the weights and links to them are below.
<div align="center">
    <image src='./pic/flow.png' width="40%">
</div>

[link](https://huggingface.co/DecoderImmortal/CDA4GEC)
|Name|Data Info|Download Link|
|:--:|--|--|
|Stage1|Pre-training on [C4 synthetic data](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction) with 200M scale|[CDA4GEC](https://huggingface.co/DecoderImmortal/CDA4GEC)/tree/main/stage1_checkpoint_best.pt|
|Stage2+|Fine-tuning on the augmented Lang8, NUCLE, FCE and W&I+L datasets|[CDA4GEC](https://huggingface.co/DecoderImmortal/CDA4GEC)/tree/main/stage2_checkpoint_best.pt|
|Stage3+|Continue fine-tuning on the augmented W&I+L dataset|[CDA4GEC](https://huggingface.co/DecoderImmortal/CDA4GEC)/tree/main/stage3_checkpoint_best.pt|

# Synthetic Data
> We only release the synthetic pseudo-data, please follow the official process to apply for the original annotated data.


|DataInfo|Amount|Source|Path|
|:--:|:--:|:--:|:--:|
|stage2+|2M|Lang-8 & NUCLE & FCE & W&I+L|[CDA4GEC](https://huggingface.co/DecoderImmortal/CDA4GEC)/tree/main/pseudo/stage2|
|stage3+|200K|W&I+L|[CDA4GEC](https://huggingface.co/DecoderImmortal/CDA4GEC)/tree/main/pseudo/stage3|

# Details
## Training Baseline Model
In this paper, we select the Fairseq framework for more efficient model training and inference.
We use the official bart-large weights and the gpt2bpe vocab for training.
1) Tokenize the raw dataset:
- ```sh ./script/tokenizer.sh```
2) Train the Bart-based GEC model:
- ```sh ./script/train.sh```

## Pattern Pool and Pesudo Data Construction

1. Error Pattern Construction
We use the ERRANT tool to extract error pattern pairs from the parallel corpus.
Subsequently, templates for synthetic data are constructed by sampling.
```shell
# Using the errant tool to extract error patterns from the parallel corpus.
python ./src/errant_tag.py --file_path ./data/annotated/train.json --output_path ./data/annotated/train_we.json

# Sampling from the error pattern pool to construct synthetic data
python ./src/get_pattern_pool.py --file_with_error ./data/annotated/train_we.json --sample_num 10000 --min_count 2 --gram 1 --output_path ./data/annotated
```
- Examples for Grammatical Error Patterns
```json
# template.json
{
  {
    "input": "had",
    "wrong": "spent"
  },
  {
    "input": "does",
    "wrong": "did"
  },
  {
    "input": "move around easily",
    "wrong": "move easily"
  },
  {
    "input": "to <mask> quickly",
    "wrong": "on <mask> rapidly"
  },
}

```

2. Pesudo Data Generation
We have tried two ways to complete the context generation of the pattern:
- Fine-tuning
```shell
python ./src/gpt_pattern_generate.py --gpt_model_path download_gpt_model_path --template_path above_template.json --output_path ./data/output
```
- In-context Learning
```shell
python ./src/gpt_pattern_generate.py --llama_model_path download_llama_model_path --template_path above_template.json --output_path ./data/output
```

## Re-label Part
After obtaining the augmented data, we relabel the output of the synthetic data with the baseline model:
- ```sh ./script/relabel.sh```
## Evaluation
We use [m2scorer_python3](https://github.com/Katsumata420/m2scorer_python3) for local metrics evaluation (P, R, F_0.5):

- ```sh ./script/evaluation.sh```

# Citation
If you find this work is useful for your research, please cite our paper:

```
@inproceedings{wang-etal-2024-improving-grammatical,
    title = "Improving Grammatical Error Correction via Contextual Data Augmentation",
    author = "Wang, Yixuan  and
      Wang, Baoxin  and
      Liu, Yijun  and
      Zhu, Qingfu  and
      Wu, Dayong  and
      Che, Wanxiang",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.647",
    pages = "10898--10910",
}
```