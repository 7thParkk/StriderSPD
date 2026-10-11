"""Command-line configuration; this module has no ML dependencies."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def cuda_devices(value):
    devices = [item.strip() for item in value.split(",")]
    if len(devices) != 3 or any(
            not item.startswith("cuda:") or not item[5:].isdigit() for item in devices):
        raise argparse.ArgumentTypeError(
            "use three CUDA roles: GNN, token embedding, adapter head; "
            "for example cuda:1,cuda:2,cuda:3")
    return devices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="StriderSPD V1 training and evaluation")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--do_train", action="store_true")
    action.add_argument("--do_test", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check manifest and input files without loading a model")
    parser.add_argument("--batch_size", type=positive_int, default=2)
    parser.add_argument("--eval_batch_size", type=positive_int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epoch", type=positive_int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--save_path", default=str(REPO_ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--root_path", default=str(REPO_ROOT / "benchmark_1720"))
    parser.add_argument("--ass_path", help="CFG JSON root; defaults to <root_path>/CFG")
    parser.add_argument("--pse_path", help="Pseudo-code JSON root; defaults to <root_path>/pseudo")
    parser.add_argument("--eval_split", default="benchmark_new", help="Load <eval_split>_infos.json")
    parser.add_argument("--expected_eval_samples", type=int, default=1720, help="Required evaluation count; 0 disables the fixed count for another dataset")
    parser.add_argument("--attention_layer_frequency", type=positive_int, default=1)
    parser.add_argument("--llm_architecture_type", choices=("qwen2", "qwen3", "llama"), default="qwen3")
    parser.add_argument("--pretrained_llm_path", help="HF repository or local model directory; paper joint training requires its LoRA-merged initializer")
    parser.add_argument("--unixcoder_path", default="microsoft/unixcoder-base", help="HF repository or local UniXcoder config/tokenizer directory")
    parser.add_argument("--checkpoint_path", help="Complete V1 state dictionary required for evaluation")
    parser.add_argument("--gnn_devices", type=cuda_devices, default=cuda_devices("cuda:1,cuda:2,cuda:3"), help="GNN, token embedding, adapter-head CUDA roles; repeats are allowed")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--log_dir", default=str(REPO_ROOT / "outputs" / "logs"))
    args = parser.parse_args(argv)
    if args.expected_eval_samples < 0:
        parser.error("--expected_eval_samples must be nonnegative")
    if args.do_test and not args.checkpoint_path:
        parser.error("--do_test requires --checkpoint_path")
    if args.do_train and not args.pretrained_llm_path:
        parser.error("--do_train requires --pretrained_llm_path for the joint-stage initializer")
    if args.pretrained_llm_path is None:
        if args.llm_architecture_type != "qwen3":
            parser.error("this architecture requires --pretrained_llm_path")
        args.pretrained_llm_path = "Qwen/Qwen3-8B"
    args.ass_path = args.ass_path or str(Path(args.root_path) / "CFG")
    args.pse_path = args.pse_path or str(Path(args.root_path) / "pseudo")
    return args
