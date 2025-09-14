import os
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from rec_attention.qwen2_model_v3 import Qwen2Model, Qwen2ModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from rec_attention.llama_model_v3 import LlamaModelForCausalLM
from rec_attention.qwen3_model import Qwen3ModelForCausalLM
from rec_attention.mistral_model import MistralForCausalLM
# from rec_attention.dscv2_model import DeepseekV2ForCausalLM
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM
def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # model = LlamaModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    
    # 加载qwen3-8b
    # model = Qwen3ModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     trust_remote_code=True
    # )

    # 加载 qwen7b、qwen0.5b
    # model = Qwen2ModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    
    # model = MistralForCausalLM.from_pretrained(
    #     model_path,
    #     device_map="auto",
    #     trust_remote_code=True
    # )

    model = DeepseekV2ForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    print(model)
    model.to(torch.bfloat16)
    config = AutoConfig.from_pretrained(model_path)
    return model, tokenizer, config