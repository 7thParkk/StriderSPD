"""Train the StriderSPD joint model or evaluate its complete paper checkpoint."""

import os
from pathlib import Path

from config.arguments import parse_args
from utils.dataset_manifest import validate_dataset


def main(argv=None):
    args = parse_args(argv)
    splits = ("train", "dev") if args.do_train else (args.eval_split,)
    for split in splits:
        expected = args.expected_eval_samples if args.do_test else None
        items = validate_dataset(args.root_path, args.ass_path, args.pse_path,
                                 split, expected_count=expected or None)
        print(f"DATA_OK split={split} samples={len(items)}")
    if args.do_test and not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(args.checkpoint_path)
    if args.preflight:
        print("PREFLIGHT_OK; model execution has not started")
        return

    # Keep --help and dataset preflight usable without the GPU dependencies.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["STRIDERSPD_LOG_DIR"] = args.log_dir
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from utils.loader import load_model_tokenizer
    from utils.seed import set_seed
    from model.base_llm import My_LLM
    from model.base_gnn import My_GNN_ManualParallel
    from model.joint_attention_model import My_Model
    from data.get_dataset import GetDataset
    from trainer.trainer import train
    from trainer.eval import evaluate

    args.n_gpu = torch.cuda.device_count()
    if not args.n_gpu:
        raise RuntimeError("StriderSPD execution requires CUDA GPUs with BF16 support.")
    for device in args.gnn_devices:
        index = int(device.split(":")[1])
        if index >= args.n_gpu:
            raise RuntimeError(
                f"{device} is unavailable with {args.n_gpu} visible GPU(s). "
                "Set --gnn_devices to three visible CUDA device roles; "
                "for one sufficiently large GPU use cuda:0,cuda:0,cuda:0."
            )
    llm_model, llm_tokenizer, llm_config = load_model_tokenizer(
        args, args.pretrained_llm_path)
    node_tokenizer = AutoTokenizer.from_pretrained(args.unixcoder_path)
    args.device = next(llm_model.parameters()).device
    args.llm_hidden_size = llm_config.hidden_size
    generation_config = GenerationConfig(
        eos_token_id=llm_tokenizer.eos_token_id,
        pad_token_id=llm_tokenizer.pad_token_id,
        max_new_tokens=4096, num_return_sequences=1, num_beams=1,
        do_sample=False, output_hidden_states=True,
        return_dict_in_generate=True,
    )
    llm = My_LLM(model=llm_model, config=llm_config,
                 tokenizer=llm_tokenizer, generation_config=generation_config)
    gnn = My_GNN_ManualParallel(
        args=args, model_name_or_path=args.unixcoder_path,
        gnn_out_dims=llm_config.hidden_size *
                     (llm_config.num_hidden_layers // args.attention_layer_frequency + 1),
        device_map=args.gnn_devices, target_device=str(args.device),
    )
    joint_model = My_Model(base_gnn=gnn, base_llm=llm, args=args)

    # Preserve the paper code's training initialization order.
    set_seed(args.seed)
    if args.do_train:
        train(args, joint_model, node_tokenizer, llm_tokenizer)
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu",
                                weights_only=True)
        joint_model.load_state_dict(checkpoint, strict=True)
        del checkpoint
        dataset = GetDataset(node_tokenizer, llm_tokenizer, args.root_path,
                             args.ass_path, args.pse_path, mode=args.eval_split)
        if args.expected_eval_samples and len(dataset) != args.expected_eval_samples:
            raise RuntimeError(f"Expected {args.expected_eval_samples} samples, got {len(dataset)}")
        evaluate(joint_model, dataset, args)


if __name__ == "__main__":
    main()
