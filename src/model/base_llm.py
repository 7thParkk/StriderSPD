import os
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from config.logger import setup_logger
logger = setup_logger()

class My_LLM(nn.Module):
    
    def __init__(self, model, tokenizer, config, generation_config):
        super(My_LLM, self).__init__()
        self.model = model
        self.device = next(model.parameters()).device  # Get the device from the model parameters
        self.config=config
        self.tokenizer=tokenizer
        self.generation_config=generation_config
        self.layer_norm = nn.LayerNorm(self.config.hidden_size).to(self.device)
    

    
    def forward(self, data, gnn_embedding, kb_config, x_kb_queries):
        
        llm_embedding = self.model(
            input_ids=data["seq_input"]["input_ids"],
            attention_mask=data["seq_input"]["attention_mask"],
            kb_kvs=gnn_embedding,
            kb_config=kb_config,
            x_kb_queries=x_kb_queries,
            output_hidden_states=True,
            return_dict=True
        )
        llm_embedding_last = llm_embedding.hidden_states[-1] # torch.Size([batch_size, seq_length, hidden_size])
        # return self.layer_norm(llm_embedding_last)
        return llm_embedding_last