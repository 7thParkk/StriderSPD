import torch
from transformers import AutoTokenizer, AutoConfig


def load_model_tokenizer(args, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if args.llm_architecture_type == "qwen3":
        from rec_attention.qwen3_model import Qwen3ModelForCausalLM as model_class
    elif args.llm_architecture_type == "qwen2":
        from rec_attention.qwen2_model_v3 import Qwen2ModelForCausalLM as model_class
    elif args.llm_architecture_type == "llama":
        from rec_attention.llama_model_v3 import LlamaModelForCausalLM as model_class
    else:
        raise ValueError(args.llm_architecture_type)
    # Keep the original V1 initialization and final BF16 model dtype.
    # For --do_test, strict loading subsequently replaces every model tensor.
    model = model_class.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True)
    model.to(torch.bfloat16)
    config = AutoConfig.from_pretrained(model_path)
    return model, tokenizer, config
