# StrideSPD
## Introduction

## Datasets
You can configure the path of the dataset through parameters `--root_path`, `--ass_path`, and `--pse_path`
## Pre-trained Model
You can configure the path of the LLM through parameter`--pretrained_llm_path`, the LLM's hidden size through parameter `--llm_hidden_size`, and the LLM's architecture type through parameter `--llm_architecture_type`
## Train
```bash
python main.py \
  --do_train \
  --eval_batch_size 2 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --epoch 8
  --pretrained_llm_path "the LLM's path you saved" \
  --llm_architecture_type "the LLM's architecture type" \

```

## Eval
```bash
python main.py \
  --do_test \
  --eval_batch_size 2 \
  --pretrained_llm_path "the LLM's architecture type" \
  --llm_architecture_type qwen3 \
  

```
