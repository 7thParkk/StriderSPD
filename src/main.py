import os
os.environ["HF_ENDPOINT"] = "https:/hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
import time
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    RobertaTokenizer
)
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from utils.loader import load_model_tokenizer
from utils.helper import RunTime
from utils.seed import set_seed
from model.base_llm import My_LLM
from model.base_gnn import My_GNN,  My_GNN_ManualParallel
from model.joint_attention_model import My_Model
from data.get_dataset import GetDataset
from trainer.trainer import train
from trainer.eval import evaluate
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from utils.collect import custom_collate_fn
from tqdm import tqdm
from utils.move_to_device import move_to_device 
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler


def main():
    llm_model_path = r"/home/res/vul/pretrained_models/qwen7b_merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/qwen3_8b_nothink_merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/llm4decompile_merge3"
    # llm_model_path = r"/home/res/vul/pretrained_models/qwen0_5b_merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/dsc-7b-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/yi-9b-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/qwen3-8b/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    # llm_model_path = r"/home/res/vul/pretrained_models/mistral-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/llama3_1-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/deepseek_1_3-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/llmcompile-7b-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/llama3.2_3b-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/yi1.5b-merge"
    # llm_model_path = r"/home/res/vul/pretrained_models/dsc16b_merge"

    pretrained_model_path = 'microsoft/unixcoder-base'

    llm_model, llm_tokenizer, llm_config = load_model_tokenizer(llm_model_path)
    node_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    
    # set device
    device = next(llm_model.parameters()).device

    generation_config = GenerationConfig(
        eos_token_id=llm_tokenizer.eos_token_id,
        pad_token_id=llm_tokenizer.pad_token_id,
        max_new_tokens=4096,
        num_return_sequences=1,
        num_beams=1,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True
    )

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--do_train", default= False, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", default= False, action='store_true',
                        help="Whether to run test.")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epoch", default=4, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="gradient_accumulation_steps.")
    parser.add_argument('--seed', type=int, default=44,
                        help="Random seed for initialization.")
    parser.add_argument('--save_path', type=str, default=r'./joint_model',
                        help="Path to save model.")
    parser.add_argument('--root_path', type=str, default=r'/home/data/',
                        help="Path to data information json file.")
    parser.add_argument('--ass_path', type=str, default=r'/home/data/@CFGs',
                        help="Path to assembly code data.")
    parser.add_argument('--pse_path', type=str, default=r'/home/data/@Pseudos',
                        help="Path to pseudo code data.")


    parser.add_argument('--attention_layer_frequency', type=int, default=1,
                        help="attention_layer_frequency")

    # Set arguments
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # load LLM & GNN
    llm = My_LLM(
        model=llm_model,
        config=llm_config,
        tokenizer=llm_tokenizer,
        generation_config=generation_config
    )
    gnn = My_GNN_ManualParallel(
        model_name_or_path=pretrained_model_path,
        gnn_out_dims=llm_config.hidden_size * (llm_config.num_hidden_layers / args.attention_layer_frequency + 1),
    )
    # gnn = gnn.to(device)

    # load joint model
    joint_model = My_Model(
        base_gnn=gnn,
        base_llm=llm,
        args=args
    )

    # Set seed
    set_seed(args.seed)

    if args.do_train:
        train(args, joint_model, node_tokenizer, llm_tokenizer)
    if args.do_test:
        weight_path = r"/home/res/vul/joint_model/model_epoch6_best_acc_qwen7b_822.pth"
        joint_model.load_state_dict(torch.load(weight_path, weights_only=True))
        eval_dataset = GetDataset(node_tokenizer, llm_tokenizer, args.root_path, args.ass_path, args.pse_path, mode="benchmark_new")
        evaluate(joint_model, eval_dataset, args)

if __name__ == '__main__':
    main()