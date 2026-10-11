# StriderSPD

Repository for the TSE paper "StriderSPD: Structure-Guided Joint Representation Learning for Binary Security Patch Detection".

## Install

Use Linux, Python 3.11, and CUDA GPUs supporting BF16. The original execution uses four visible GPUs in one Python process; the code uses model parallelism.

```bash
git clone https://github.com/7thParkk/StriderSPD.git
cd StriderSPD
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment pins PyTorch 2.4.0 with CUDA 12.1, Transformers 4.55.0, Accelerate 1.10.0 and PyG 2.6.1. Allow disk space for the checkpoint and base-model files, and RAM for loading the full checkpoint on CPU.

## Download the trained model

The paper joint checkpoint is hosted at [hanjin996/StriderSPD](https://huggingface.co/hanjin996/StriderSPD):

- [model_epoch5_best_acc_qwen3_8b_824.pth](https://huggingface.co/hanjin996/StriderSPD/resolve/main/model_epoch5_best_acc_qwen3_8b_824.pth), 25,682,301,190 bytes (25.68 GB).
- [benchmark_1720.zip](https://huggingface.co/hanjin996/StriderSPD/resolve/main/benchmark_1720.zip), an alternative download of the benchmark already included here.

Download and verify the checkpoint:

```bash
python scripts/download_checkpoint.py --output-dir checkpoints
```

SHA256: `991045a6bd0c8a30b113b1d2f633ed6d464f48ec06f4325b6adfef27db5e01e7`.

The checkpoint stores the complete joint model, including the Qwen3-8B backbone, UniXcoder node encoder, GNN, attention and classifier. Evaluation uses the public `Qwen/Qwen3-8B` and `microsoft/unixcoder-base` assets to initialize the custom architecture and tokenizers, then strictly replaces all model tensors with the checkpoint. Local model directories can be supplied with `--pretrained_llm_path` and `--unixcoder_path`.

## Benchmark

```text
benchmark_1720/
├── benchmark_new_infos.json
├── CFG/<project>/<optimization>/<file>.json
└── pseudo/<project>/<optimization>/<file>.json
```

There are 1,720 paired samples: 1,010 security and 710 non-security. Each modality has 3,440 pre/post JSON files. Projects, optimization levels, file names, manifest order and JSON contents are preserved from the original benchmark. CFG JSON contains nodes and edges; pseudo JSON contains a `pseudo_code` list. Labels and pair paths are in the manifest.

Check all data representations without ML dependencies:

```bash
python scripts/check_benchmark.py
```

When using the ZIP download, extract it into `benchmark_1720/`:

```bash
unzip benchmark_1720.zip -d benchmark_1720
```

## Inference / evaluation

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py \
  --do_test \
  --checkpoint_path checkpoints/model_epoch5_best_acc_qwen3_8b_824.pth \
  --llm_architecture_type qwen3 \
  --attention_layer_frequency 1 \
  --root_path benchmark_1720 \
  --ass_path benchmark_1720/CFG \
  --pse_path benchmark_1720/pseudo \
  --eval_batch_size 1 \
  --output_dir outputs/eval
```

Add `--preflight` to validate the manifest, all required files and checkpoint path before loading a model. Evaluation requires all 1,720 samples and fails when input files are missing. Results are saved to `outputs/eval/predictions.jsonl` in manifest order and `outputs/eval/summary.json`. Positive means a security patch; prediction uses `sigmoid(logit) >= 0.5`.

The three GNN roles default to `cuda:1,cuda:2,cuda:3`. To run with one sufficiently large GPU, use `CUDA_VISIBLE_DEVICES=0` and `--gnn_devices cuda:0,cuda:0,cuda:0`; memory sufficiency and GPU result parity for this layout have not been measured here. Use `python src/main.py --help` for the complete CLI.

For another evaluation set, provide the same manifest/representation schema, use `--eval_split <name>` to load `<name>_infos.json`, and set `--expected_eval_samples <count>`.

## Joint training

The joint-training entrypoint expects a prepared training dataset and a Qwen3-8B initializer. The paper run initialized from a Qwen3-8B model with its semantic fine-tuning LoRA weights already merged. Supply that local merged model to reproduce the joint-training configuration.

```text
training_data/
├── train_infos.json
├── dev_infos.json
├── CFG/<project>/<optimization>/<file>.json
└── pseudo/<project>/<optimization>/<file>.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py \
  --do_train \
  --pretrained_llm_path /path/to/qwen3-8b-lora-merged \
  --llm_architecture_type qwen3 \
  --attention_layer_frequency 1 \
  --root_path /path/to/training_data \
  --ass_path /path/to/training_data/CFG \
  --pse_path /path/to/training_data/pseudo \
  --batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --epoch 8 \
  --learning_rate 1e-5 \
  --weight_decay 1e-4 \
  --adam_epsilon 1e-6 \
  --max_grad_norm 1.0 \
  --seed 44 \
  --save_every_epoch \
  --save_path outputs/checkpoints
```

The 1,720-sample release supplies evaluation data. The train/dev representations and semantic-stage merged initializer must be prepared separately. Training selects the best validation-accuracy checkpoint and records its path in `best_checkpoint.txt`; `--save_every_epoch` also retains each epoch checkpoint.

## Paper reference

Paper checkpoint reference results on the 1,720-sample benchmark:

| Samples | Accuracy | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- |
| 1,720 | 0.854070 | 0.822979 | 0.957426 | 0.885126 | 0.292958 |
