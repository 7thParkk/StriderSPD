import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from model.cross_attention import CrossAttention
from config.logger import setup_logger
from accelerate import dispatch_model, infer_auto_device_map
from rec_attention.kblam_config import KBLaMConfig
logger = setup_logger()

class My_Model(nn.Module):
    def __init__(self, base_gnn, base_llm, args):
        super().__init__()
        
        self.llm = base_llm
        self.gnn = base_gnn

        self.kb_config = KBLaMConfig(kb_layer_frequency=args.attention_layer_frequency)

        self.device = next(base_llm.parameters()).device

        self.norm = nn.LayerNorm(base_llm.config.hidden_size).to(self.device)

        self.cross_attention = CrossAttention(base_llm.config.hidden_size).to(self.device)

        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(base_llm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)

        self.dropout = nn.Dropout(0.2)

        # 初始化cross_attention和分类器的权重
        for module in [self.cross_attention, self.mlp]:
            module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.01)

    def forward(self, data):

        # graph-branch输出的内容，我用_替代了没用的x_key_graph和x_value_graph
        x_query, x_key, x_value, _, _, x_kb_queries = self.gnn(data)


        # 将图信息注入LLM-branch
        llm_embedding = self.llm(
            data=data,
            gnn_embedding=(x_key, x_value),
            kb_config=self.kb_config,
            x_kb_queries=x_kb_queries
        ).to("cuda:0")

        x_query = x_query.to("cuda:0")
        x_key = x_key.to("cuda:0")
        x_value = x_value.to("cuda:0")

        # 得到用于2分类的vector
        # embedding = llm_embedding
        embedding = self.cross_attention(x_query, llm_embedding)
        embedding = self.dropout(embedding)

        # 进行二分类
        output = torch.mean(embedding, dim = 1)
        # batch_size, hidden_size
        output = self.norm(output)
        logits = self.mlp(output)

        return logits

    def to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(v, device) for v in batch)
        elif hasattr(batch, "to"):
            return batch.to(device)
        else:
            return batch