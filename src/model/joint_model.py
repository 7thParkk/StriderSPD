import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from model.cross_attention import CrossAttention
from config.logger import setup_logger
from accelerate import dispatch_model, infer_auto_device_map
logger = setup_logger()

class My_Model(nn.Module):
    def __init__(self, base_gnn, base_llm):
        super().__init__()
        self.llm = base_llm
        # self.gnn = base_gnn
        self.gnn_device_map = infer_auto_device_map(
            base_gnn,
            max_memory={i: "10GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["GatedGCNKernel"]  # 禁止分割的模块类名
        )
        self.gnn = dispatch_model(
            base_gnn,
            device_map=self.gnn_device_map
        )

        self.device = next(base_llm.parameters()).device
        self.cross_attention = CrossAttention(base_llm.config.hidden_size)
        self.cross_attention = CrossAttention(base_llm.config.hidden_size).to(self.device)
        # self.cross_attention = self.cross_attention.to(dtype=torch.float16)
        self.norm = nn.LayerNorm(base_llm.config.hidden_size)
        self.norm = nn.LayerNorm(base_llm.config.hidden_size).to(self.device)

        # 多层感知机，用于对数据输入进行非线性变化和降维
        self.mlp = nn.Sequential(
            nn.Linear(base_llm.config.hidden_size, 128),    
            nn.LeakyReLU(negative_slope=5e-2),             
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)
        # self.mlp = self.mlp.to(dtype=torch.float16)
        self.dropout = nn.Dropout(0.2)
        for module in [self.cross_attention, self.mlp]:
            module.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.05))
            nn.init.constant_(m.bias, 0.01)

    def forward(self, data):
        llm_embedding = self.llm(data)
        gnn_embedding = self.gnn(data)
        embedding = self.cross_attention(gnn_embedding, llm_embedding)
        embedding = self.dropout(embedding)
        output = torch.mean(embedding, dim = 1)
        # batch_size, hidden_size(4096)
        output = output.squeeze(dim = 1)
        output = self.norm(output)
        logits = self.mlp(output)
        return logits

