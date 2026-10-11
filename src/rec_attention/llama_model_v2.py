import math
import os
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
logger = logging.get_logger(__name__)
PADDING_VALUE = torch.finfo(torch.bfloat16).min
from rec_attention.kblam_config import KBLaMConfig
from transformers.generation import GenerationMixin

class RectangularAttention(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
         # 执行层索引
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        # hidden_size 应被 num_heads 整除
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # key 和 value 的头数
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # RopE（旋转位置编码）的参数
        self.rope_theta = config.rope_theta

        # 是否是因果注意力，这里我们设置为 False, 默认为 False
        self.is_causal = True
        self.score_shift = nn.Parameter(torch.zeros(self.num_heads, 1) - 3)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_proj_new = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )
        self._init_rope()

    # 根据 self.config 的参数来初始化不同类型的旋转位置编码，旋转位置编码用于在 q、k 中保持位置信息
    # 这里 可能是 transformer 代码更新了，不需要手动根据各种类型来初始化
    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(self.config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        kb_kvs_graph: Optional[tuple] = None,
        save_attention_weights: bool = True,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states_2 = self.q_proj_new(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape
        # query_states：[batch_size, num_heads, q_len, head_dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        query_states_2 = query_states_2.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # 位置编码
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 通过这种方式，模型可以在处理序列数据时利用过去的键值对缓存，从而加速解码过程
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # repeat_kv：将key_states 和value_states 复制num_key_value_groups 次
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 每多少层会使用 graph
        kb_layer_frequency = kb_config.kb_layer_frequency
        
        attn_weights_2 = None
        # 注意，我们的框架下，kb_kvs是 gnn 处理得到的 tensor
        if kb_kvs is not None:
            # 判断当前层是否需要 merge graph信息
            if self.layer_idx % kb_layer_frequency == 0:
                
                # 注意，目前我们的做法是 key，value 都赋为graph处理后得到的结果
                if self.layer_idx < 50:
                    kb_keys, kb_values = (
                        kb_kvs  # (kb_len, head_dim * num_heads * num_adapters)
                    )
                else :
                    kb_keys, kb_values = (
                        kb_kvs_graph  # (kb_len, head_dim * num_heads * num_adapters)
                    )
                kb_idx = (
                    self.layer_idx // kb_layer_frequency
                )
                # kb_keys [bsz, kb_len, head_dim * num_heads * num_adapters], 其中，gnn 处理后，kb_len = 1    
                kb_len = kb_keys.shape[1]
                # 将kb_keys 分成1 + self.config.num_hidden_layers // kb_layer_frequency个 block
                kb_keys = kb_keys.view(
                    bsz,
                    kb_len,
                    1 + self.config.num_hidden_layers // kb_layer_frequency,
                    -1,
                )[:, :, kb_idx]
                kb_values = kb_values.view(
                        bsz,
                        kb_len,
                        1 + self.config.num_hidden_layers // kb_layer_frequency,
                        -1,
                )[:, :, kb_idx]
                kb_keys = kb_keys.view(
                    bsz, kb_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                kb_values = kb_values.view(
                    bsz, kb_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                
                # 拼接kb 的 key 和 value 到 hidden_states 的 key 和 value
                # (bsz, num_heads, kb_len + q_len, head_dim)
                key_states = torch.concat([kb_keys, key_states], dim=2)
                value_states = torch.concat([kb_values, value_states], dim=2)
                # 更新 mask
                kb_len = kb_keys.shape[2]
                kb_atten_mask = attention_mask.new_zeros(bsz, 1, q_len, kb_len)
                padding_mask = torch.all(
                    attention_mask < 0, -1, keepdim=True
                )  # (bsz, num_heads, q_len, 1)
                kb_atten_mask = (
                    padding_mask * PADDING_VALUE + (~padding_mask) * kb_atten_mask
                )
                attention_mask = torch.concat([kb_atten_mask, attention_mask], dim=-1)
                # # 打印维度确认
                # print("Final attention_mask shape:", attention_mask.shape)
                # # 提取维度信息
                # bsz, num_heads, q_len, total_len = attention_mask.shape
                # kb_len = kb_keys.shape[2]

                # # 取出 KB 部分的 attention mask
                # kb_mask = attention_mask[:, :, :, :kb_len]  # shape: [bsz, num_heads, q_len, kb_len]
                # print("KB attention mask (before softmax):")
                # print(kb_mask)

                # # 取出原始 query 部分对应的 mask
                # query_mask = attention_mask[:, :, :, kb_len:]  # shape: [bsz, num_heads, q_len, seq_len]
                # print("Query attention mask (before softmax):")
                # print(query_mask)


        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        # 是否开启kb 与 主序列分离的 query head，是否对 kb 部分和主序列部分的attention 分数分别计算
        sep_query_head = kb_config.sep_query_head
        kb_scale_factor = kb_config.kb_scale_factor
        if sep_query_head:
            if kb_kvs is not None:
                if self.layer_idx % kb_layer_frequency == 0:
                    # If we have pruned the KB tokens, then this quantity should have been computed,
                    # if not, then we compute it here
                    if attn_weights_2 is None:
                        attn_weights_2 = torch.matmul(
                            query_states_2, kb_keys.transpose(2, 3)
                        ) / math.sqrt(self.head_dim)
                    attn_weights = attn_weights[:, :, :, kb_len:]
                    if kb_scale_factor is not None:
                        attn_weights_2 = (
                            attn_weights_2 - np.log(kb_len) + np.log(kb_scale_factor)
                        )
                    attn_weights = torch.concat([attn_weights_2, attn_weights], -1)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # 保存weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if not attn_weights.requires_grad:
            # TODO: Make this function injectable
            if save_attention_weights:
                if q_len > 1:
                    save_path = os.path.join(
                        attention_save_loc,
                        f"{attention_file_base_name}_{self.layer_idx}.npy",
                    )
                    np.save(
                        save_path,
                        attn_weights.to(torch.float32).cpu().detach().numpy(),
                    )
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # attn_output: [batch_size, query_size, hidden_size]
        return attn_output, attn_weights, past_key_value
    

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = RectangularAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        kb_kvs: Optional[tuple] = None,
        kb_config: Optional[KBLaMConfig] = None,
        kb_kvs_graph: Optional[tuple] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Rectangular Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            kb_kvs=kb_kvs,
            kb_kvs_graph=kb_kvs_graph,
            kb_config=kb_config,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        kb_kvs: Optional[tuple] = None,
        kb_kvs_graph: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        kb_config: Optional[KBLaMConfig] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(
            past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    kb_kvs,
                    kb_config,
                    save_attention_weights,
                    attention_save_loc,
                    attention_file_base_name,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    kb_kvs=kb_kvs,
                    kb_kvs_graph=kb_kvs_graph,
                    kb_config=kb_config,
                    save_attention_weights=save_attention_weights,
                    attention_save_loc=attention_save_loc,
                    attention_file_base_name=attention_file_base_name,
                )
            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        
        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask

class LlamaModelForCausalLM(LlamaPreTrainedModel):

    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        # base_model_name_or_path = (
        #     config.base_model_name_or_path
        #     if hasattr(config, "base_model_name_or_path")
        #     else config._name_or_path
        # )
        # self.model = LlamaModel.from_pretrained(
        #     base_model_name_or_path, torch_dtype=config.torch_dtype
        # )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            self.model.config.hidden_size, self.model.config.vocab_size, bias=False
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def get_config(self):
        return self.config

    def set_config(self, config):
        self.config = config

    def update_generation_config(self, tokenizer):
        self.generation_config.pad_token_id = tokenizer.pad_token_id
        self.generation_config.eos_token_id = tokenizer.eos_token_id
    
    def load_query_head(self, ckpt_dir):
        learned_query_heads = torch.load(ckpt_dir)
        assert len(learned_query_heads) == self.model.config.num_hidden_layers
        for i, attn_layer in enumerate(self.model.layers):
            attn_layer.self_attn.q_proj_new.load_state_dict(
                learned_query_heads[f"layer_{i}"]
            )
        self.config.sep_query_head = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        kb_kvs: Optional[tuple] = None,
        kb_kvs_graph: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        kb_config: Optional[KBLaMConfig] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            kb_kvs=kb_kvs,
            kb_kvs_graph=kb_kvs_graph,
            kb_config=kb_config,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )