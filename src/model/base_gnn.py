import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from transformers import RobertaModel, RobertaConfig
import lightning as L
from torch_geometric.nn.models.basic_gnn import BasicGNN
from model.gsa_pool import GSAPool
from config.logger import setup_logger
logger = setup_logger()


class GatedGCNKernel(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dims):
        super(GatedGCNKernel, self).__init__(aggr='add')
        self.A = nn.Linear(in_channels, out_channels)
        self.B = nn.Linear(in_channels, out_channels)
        self.C = nn.Linear(edge_dims, out_channels)
        self.D = nn.Linear(in_channels, out_channels)
        self.E = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x = (x, x) # j -> i

        Ax = self.A(x[1])

        Dx = self.D(x[1])
        
        Bx = self.B(x[0]) # j

        Ex = self.E(x[0]) # j
        
        Ce = self.C(edge_attr)

        out = self.propagate(edge_index, B=Bx, D=Dx, E=Ex, edge_attr=Ce)
        
        out = out + Ax # Ax是At-1*ht-1
        # out = F.relu(out + Ax)
        return out

    def message(self, B_j, D_i, E_j, edge_attr):
        gate = torch.sigmoid(D_i + E_j + edge_attr)
        return gate * B_j


class GatedGCN_EKernel(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dims):
        super().__init__(aggr='add')
        self.A = nn.Linear(in_channels, out_channels)
        self.B = nn.Linear(in_channels, out_channels)
        self.C = nn.Linear(edge_dims, out_channels)
        self.D = nn.Linear(in_channels, out_channels)
        self.E = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, torch.Tensor):
            x = (x, x) # j -> i
        Ax = self.A(x[1])
        Dx = self.D(x[1])
        
        Bx = self.B(x[0])
        Ex = self.E(x[0])
        
        Ce = self.C(edge_attr)
        
        edge_attr = self.edge_updater(edge_index=edge_index, D=Dx, E=Ex, edge_attr=Ce)
        out = self.propagate(edge_index, B=Bx, edge_attr=edge_attr)
        out = out + Ax
        
        return out, edge_attr

    def message(self, B_j, edge_attr):
        gate = torch.sigmoid(edge_attr)
        return gate * B_j
        
    def edge_update(self, D_i, E_j, edge_attr):
        return D_i + E_j + edge_attr
    

class GatedGCN(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True
    def init_conv(self, in_channels: int, out_channels: int, edge_dims: int, **kwargs):
        return GatedGCNKernel(in_channels, out_channels, edge_dims=edge_dims)

class My_GNN(L.LightningModule):
    # embedding_dims 大小等于 unixcoder的hidden size
    def __init__(
        self,
        model_name_or_path='microsoft/unixcoder-base',
        num_edge_type=2,
        embedding_dims=768,
        gnn_name='gatedgcn',
        llm_hidden_size=4096,
        gnn_out_dims=896,
        gnn_layers=4,
        dropout=0.2
        # lr=1e-4,
        # milestones=[30, 60]
    ):
        super(My_GNN, self).__init__()
        
        # transformer layer用UniXcoder
        self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
        self.node_edge_embedding_size = self.unixcoder_config.hidden_size

        self.trans_encoder = RobertaModel(self.unixcoder_config)
        # 每个token的正常的embedding
        self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0)
        
        # 总共有两种边
        self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims)

        # GatedGCN用来处理被Unixcoder向量化后的CFG信息
        if gnn_name == 'gatedgcn':
            self.gnn = GatedGCN(
                in_channels=embedding_dims, 
                hidden_channels=embedding_dims, 
                num_layers=gnn_layers,
                out_channels=embedding_dims, 
                dropout=dropout, 
                jk='cat', 
                edge_dims=embedding_dims
            )
        self.pool = GSAPool(in_channels = embedding_dims)

        # self.pooling = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))
        self.layer_norm = nn.LayerNorm(gnn_out_dims)
        
        
        # 对齐LLM的hidden dim，方便后续做cross attention
        self.fc_query = nn.Linear(4*embedding_dims, llm_hidden_size)
        self.query_layernorm = nn.LayerNorm(llm_hidden_size)
        self.fc_key = nn.Linear(4*embedding_dims, gnn_out_dims)
        self.key_layernorm = nn.LayerNorm(gnn_out_dims)
        self.fc_value = nn.Linear(4*embedding_dims, gnn_out_dims)
        self.value_layernorm = nn.LayerNorm(gnn_out_dims)

    def forward(self, data): 
        # 获取数据
        a_nodes = data['a_nodes']
        a_nodes_attention_mask = data['a_nodes_attention_mask']
        a_edges = data['a_edges']
        a_edges_type = data['a_edges_type']
        b_nodes = data['b_nodes']
        b_nodes_attention_mask = data['b_nodes_attention_mask']
        b_edges = data['b_edges']
        b_edges_type = data['b_edges_type']
        
        # 打印批次中的唯一值
        print("a_batch unique:", torch.unique(data['a_batch']))  # 应输出 [0, 1, ..., batch_size-1]
        print("b_batch unique:", torch.unique(data['b_batch']))

        # 对节点进行token嵌入
        a_nodes = self.token_embedding(a_nodes)
        # 对节点进行transformer编码
        a_nodes = self.trans_encoder(inputs_embeds=a_nodes, attention_mask=a_nodes_attention_mask).pooler_output # 尝试换成Qwen3embedding
        # 对节点进行token嵌入
        b_nodes = self.token_embedding(b_nodes)
        # 对节点进行transformer编码
        b_nodes = self.trans_encoder(inputs_embeds=b_nodes, attention_mask=b_nodes_attention_mask).pooler_output # 把encoder的最后一层向量x经过一层pooler


        # 对边类型进行嵌入
        a_edges_type = self.edge_embedding(a_edges_type)
        b_edges_type = self.edge_embedding(b_edges_type)

        # 原始gnn 的方法
        # 把x和edge一起送入GNN model
        x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
        x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

        print("x1.shape() after gnn:" + str(x1.shape))
        # print("x2.shape() after gnn:" + str(x2.shape))

        x1 = torch.cat([global_mean_pool(x1, data['a_batch']), global_max_pool(x1, data['a_batch'])], dim=1)
        x2 = torch.cat([global_mean_pool(x2, data['b_batch']), global_max_pool(x2, data['b_batch'])], dim=1)  # [batch_size, 2*hidden_channels]
        x = torch.cat([x1, x2], dim=1)  # [batch_size, 4*hidden_channels]

        print("x1.shape() after cat:" + str(x1.shape))
        # print("x2.shape() after cat:" + str(x2.shape))

        x = F.dropout(x, p=0.2, training=self.training)
        # x = nn.functional.normalize(x, p=2, dim=1)
        x = x.unsqueeze(1)

        x_key = self.fc_key(x)
        x_value = self.fc_value(x)
        x_query = self.fc_query(x)
        # x = nn.functional.normalize(x, p=2, dim=1)
        
        x_key = self.key_layernorm(x_key)
        x_value = self.value_layernorm(x_value)
        x_query = self.query_layernorm(x_query)
        
        return x_query, x_key, x_value

        # 尝试 1：以 batch 中节点数量最少的 a_node+b_node 来截断
        # 1. 得到 GNN 输出
        # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)  # [num_nodes_a, dims]
        # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)  # [num_nodes_b, dims]

        # a_batch = data['a_batch']  # [num_nodes_a]
        # b_batch = data['b_batch']  # [num_nodes_b]

        # bsz = a_batch.max().item() + 1  # 假设 batch 是 0-based

        # x_list = []

        # min_node_num = float('inf')

        # # 2. 第一次遍历，找出每个图节点数并统计最小值
        # for i in range(bsz):
        #     num_nodes_i = (a_batch == i).sum().item() + (b_batch == i).sum().item()
        #     if num_nodes_i < min_node_num:
        #         min_node_num = num_nodes_i

        # # 3. 第二次遍历，拼接每个样本的节点，并截断到最小节点数
        # for i in range(bsz):
        #     x1_i = x1[a_batch == i]  # [node_numsa_i, dims]
        #     x2_i = x2[b_batch == i]  # [node_numsb_i, dims]
        #     x_i = torch.cat([x1_i, x2_i], dim=0)  # [node_numsa_i + node_numsb_i, dims]

        #     # 截断前 min_node_num 个节点（保持每个 batch 一样长）
        #     x_i = x_i[:min_node_num]  # [min_node_num, dims]

        #     x_list.append(x_i)

        # # 4. 拼成统一 batch tensor
        # x = torch.stack(x_list, dim=0)  # [bsz, min_node_num, dims]

        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.fc(x)
        # # x = nn.functional.normalize(x, p=2, dim=1)
        # x = self.layer_norm(x)
        # print("x shape" + str(x.shape))


        # return x 

        # 尝试 2：topk，然后按最短的截断
        # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
        # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

        # a_batch = data['a_batch']  # shape: [num_total_nodes_a]
        # b_batch = data['b_batch']  # shape: [num_total_nodes_b]

        # batch_size_a = a_batch.max().item() + 1
        # batch_size_b = b_batch.max().item() + 1

        # # 统计每个样本a图的节点数
        # a_nodes_count = torch.bincount(a_batch, minlength=batch_size_a)  # tensor长度是batch_size_a
        # # 统计每个样本b图的节点数
        # b_nodes_count = torch.bincount(b_batch, minlength=batch_size_b)
        
        # x1_pool, _, _, batch_a_pool, perm_a = self.pool(
        #     x1, a_edges, edge_attr=a_edges_type, batch=data['a_batch']
        # )

        # # 图 b 的池化
        # x2_pool, _, _, batch_b_pool, perm_b = self.pool(
        #     x2, b_edges, edge_attr=b_edges_type, batch=data['b_batch']
        # )

        # # 记录每个样本选中的节点数
        # num_nodes_list = []
        # num_nodes_list.append(50)

        # all_samples_nodes = []
        # batch_size = data['a_batch'].max().item() + 1
        # for i in range(batch_size):
        #     a_nodes_i = x1_pool[batch_a_pool == i]  # [num_a_i, hidden_dim]
        #     b_nodes_i = x2_pool[batch_b_pool == i]  # [num_b_i, hidden_dim]

        #     all_nodes = torch.cat([a_nodes_i, b_nodes_i], dim=0)  # [num_a_i + num_b_i, hidden_dim]
        #     num_nodes_list.append(all_nodes.size(0))
        #     all_samples_nodes.append(all_nodes)

        # # 对齐：找出最小的节点数
        # min_nodes = min(num_nodes_list)

        # # 截断对齐
        # for i in range(batch_size):
        #     all_samples_nodes[i] = all_samples_nodes[i][:min_nodes]

        # # 最终拼接：[batch_size, min_nodes, hidden_dim]
        # x = torch.stack(all_samples_nodes, dim=0)

        # x = F.dropout(x, p=0.2, training=self.training)

        # x_key = self.fc_key(x)
        # x_value = self.fc_value(x)
        # x_query = self.fc_query(x)

        # x_key = self.key_layernorm(x_key)
        # x_value = self.value_layernorm(x_value)
        # x_query = self.query_layernorm(x_query)

        # return x_query, x_key, x_value


        # # x = nn.functional.normalize(x, p=2, dim=1)
        
        # x_key = self.layer_norm(x_key)
        # x_value = self.layer_norm(x_value)
        
        # return (x_key, x_value)

        #尝试3： 2张图分别池化
        # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
        # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

        # x1_mean = global_mean_pool(x1, data['a_batch']).unsqueeze(1)
        # x2_mean = global_mean_pool(x2, data['b_batch']).unsqueeze(1)
        # x1_max = global_max_pool(x1, data['a_batch']).unsqueeze(1)
        # x2_max = global_max_pool(x2, data['b_batch']).unsqueeze(1)

        # x = torch.cat([
        #     x1_mean,   # [bsz, 1, d]
        #     x1_max,   # [bsz, 1, d]
        #     x2_mean,    # [bsz, 1, d]
        #     x2_max     # [bsz, 1, d]
        # ], dim=1) 

        # x = F.dropout(x, p=0.2, training=self.training)

        # x_key = self.fc_key(x)
        # x_value = self.fc_value(x)

        # # x = nn.functional.normalize(x, p=2, dim=1)
        
        # x_key = self.key_layernorm(x_key)
        # x_value = self.value_layernorm(x_value)
        
        # return (x_key, x_value)

# class My_GNN_ManualParallel(nn.Module):
#     def __init__(
#         self,
#         args,
#         model_name_or_path='microsoft/unixcoder-base',
#         num_edge_type=2,
#         embedding_dims=768,
#         gnn_name='gatedgcn',
#         gnn_out_dims=896,
#         gnn_layers=4,
#         dropout=0.2,
#         device_map=None,   # 新增：指定模块设备分配，如 ["cuda:1","cuda:2","cuda:3"]
#         target_device="cuda:0"  # 最终与LLM对接的设备
#     ):
#         super().__init__()
#         # 设备分配默认值（如果没传 device_map）
#         if device_map is None:
#             # token/transformer -> cuda:1, gnn -> cuda:2, pool+fc -> cuda:3
#             device_map = ["cuda:1", "cuda:2", "cuda:3"]
#         assert len(device_map) >= 3, "device_map 至少需要三个设备: [embed_dev, gnn_dev, head_dev]"

#         self.embed_device = torch.device(device_map[1])
#         self.gnn_device = torch.device(device_map[0])
#         self.head_device = torch.device(device_map[2])
#         self.target_device = torch.device(target_device)

#         # transformer layer 用 UniXcoder
#         self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
#         self.node_edge_embedding_size = self.unixcoder_config.hidden_size

#         # 把较重但可以集中的模块放到 embed_device
#         self.trans_encoder = RobertaModel(self.unixcoder_config).to(self.gnn_device)
#         self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0).to(self.embed_device)
#         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims).to(self.gnn_device)

#         # GatedGCN 放到 gnn_device
#         if gnn_name == 'gatedgcn':
#             # 假设 GatedGCN 构造函数返回一个 nn.Module，可以 .to(device)
#             self.gnn = GatedGCN(
#                 in_channels=embedding_dims,
#                 hidden_channels=embedding_dims,
#                 num_layers=gnn_layers,
#                 out_channels=embedding_dims,
#                 dropout=dropout,
#                 jk='cat',
#                 edge_dims=embedding_dims
#             ).to(self.gnn_device)
#         else:
#             raise NotImplementedError("目前仅支持 gatedgcn")

#         self.layer_norm = nn.LayerNorm(gnn_out_dims).to(self.head_device)

#         # 对齐 LLM 的 hidden dim, 放 head_device（最后结果再搬到 target_device）
#         # self.fc_query = nn.Linear(embedding_dims, llm_hidden_size).to(self.head_device)
#         # self.query_layernorm = nn.LayerNorm(llm_hidden_size).to(self.head_device)
#         # Key: Linear -> ReLU -> Linear -> LayerNorm
#         # self.fc_key = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)
#         # self.fc_value = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)

#         # Q K V Adapters
#         self.fc_query_adapter = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 32, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 32, args.llm_hidden_size, dtype=torch.bfloat16),
#             nn.LayerNorm(args.llm_hidden_size)
#         ).to(self.head_device)

#         self.fc_key= nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 32, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 32, gnn_out_dims, dtype=torch.bfloat16),
#             nn.LayerNorm(gnn_out_dims)
#         ).to(self.head_device)

#         self.fc_value = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 32, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 32, gnn_out_dims, dtype=torch.bfloat16)
#         ).to(self.head_device)

#         # Gate
#         self.gate = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 32),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 32, 1),
#             nn.Sigmoid()
#         ).to(self.head_device)


#         # self.q_adapter_from_kb = nn.Linear(llm_hidden_size, llm_hidden_size, bias=False)

#     def to_device(self, obj, device):
#         """递归地把 obj（tensor/dict/list/tuple）搬到 device 上"""
#         if isinstance(obj, dict):
#             return {k: self.to_device(v, device) for k, v in obj.items()}
#         elif isinstance(obj, (list, tuple)):
#             # 保持原类型
#             return type(obj)(self.to_device(v, device) for v in obj)
#         elif hasattr(obj, "to"):
#             return obj.to(device)
#         else:
#             return obj

    
#     def node_matching(self, x1, x2, threshold=0.80):
#         if x1.numel() == 0 or x2.numel() == 0:
#             return [], list(range(x1.size(0))), list(range(x2.size(0)))

#         sim_matrix = F.cosine_similarity(
#             x1.unsqueeze(1),  # [N1, 1, d]
#             x2.unsqueeze(0),  # [1, N2, d]
#             dim=-1
#         )  # [N1, N2]

#         cost_matrix = 1 - sim_matrix.detach().cpu().numpy()
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         matches, diffs_pre, diffs_post = [], [], []
#         for i, j in zip(row_ind, col_ind):
#             sim = sim_matrix[i, j].item()
#             if sim >= threshold:
#                 matches.append((i, j, sim))
#             else:
#                 diffs_pre.append(i)
#                 diffs_post.append(j)

#         matched_pre = {i for i, _, _ in matches}
#         matched_post = {j for _, j, _ in matches}

#         diffs_pre.extend([i for i in range(x1.size(0)) if i not in matched_pre])
#         diffs_post.extend([j for j in range(x2.size(0)) if j not in matched_post])

#         return matches, diffs_pre, diffs_post

#     def forward(self, data):
#         # ---------------------------
#         # 1) 把需要的输入先搬到 embed_device
#         # ---------------------------
#         # data 的字段可能是 tensor 或者其他容器，我们单独把需要的 tensors 搬过去
#         a_nodes = data['a_nodes'].to(self.embed_device)
#         a_nodes_attention_mask = data['a_nodes_attention_mask'].to(self.gnn_device)
#         a_edges = data['a_edges']  # edge index 通常是 LongTensor, 视实现而定
#         a_edges_type = data['a_edges_type'].to(self.gnn_device)

#         b_nodes = data['b_nodes'].to(self.embed_device)
#         b_nodes_attention_mask = data['b_nodes_attention_mask'].to(self.gnn_device)
#         b_edges = data['b_edges']
#         b_edges_type = data['b_edges_type'].to(self.gnn_device)

#         # 批次索引也可能在 cpu，需要明确搬到相应设备用于 pool（pool 会在 head_device 上执行）
#         a_batch = data['a_batch']  # 这是 batch idx，用于 pooling，最终会被搬到 head_device
#         b_batch = data['b_batch']

#         # ---------------------------
#         # 2) embedding + transformer 在 embed_device 上计算
#         # ---------------------------
#         a_nodes_emb = self.token_embedding(a_nodes)   # already on embed_device
#         a_encoded = self.trans_encoder(inputs_embeds=a_nodes_emb.to(self.gnn_device), attention_mask=a_nodes_attention_mask).pooler_output
#         # a_encoded: [N_a_tokens(number of a graph's nodes?), hidden] or [batch_size_nodes, hidden] 取决于你的输入结构

#         b_nodes_emb = self.token_embedding(b_nodes)
#         b_encoded = self.trans_encoder(inputs_embeds=b_nodes_emb.to(self.gnn_device), attention_mask=b_nodes_attention_mask).pooler_output

#         a_edges_type_emb = self.edge_embedding(a_edges_type)
#         b_edges_type_emb = self.edge_embedding(b_edges_type)

#         # ---------------------------
#         # 3) 把需要给 GNN 的张量搬到 gnn_device
#         #    注意 edge_index (a_edges/b_edges) 也要搬
#         # ---------------------------
#         a_encoded = a_encoded.to(self.gnn_device)
#         b_encoded = b_encoded.to(self.gnn_device)
#         a_edges = a_edges.to(self.gnn_device)
#         b_edges = b_edges.to(self.gnn_device)
#         a_edges_type_emb = a_edges_type_emb.to(self.gnn_device)
#         b_edges_type_emb = b_edges_type_emb.to(self.gnn_device)

#         # ---------------------------
#         # 4) 在 gnn_device 上运行 GNN（大部分计算发生在这里）
#         # ---------------------------
#         x1 = self.gnn(a_encoded, a_edges, edge_attr=a_edges_type_emb)
#         x2 = self.gnn(b_encoded, b_edges, edge_attr=b_edges_type_emb)

#         a_batch = a_batch.to(self.head_device)
#         b_batch = b_batch.to(self.head_device)

#         x1 = x1.to(self.head_device)
#         x2 = x2.to(self.head_device)
        
#         x1_mean = global_mean_pool(x1, a_batch).unsqueeze(1)
#         x2_mean = global_mean_pool(x2, b_batch).unsqueeze(1)
#         x1_max = global_max_pool(x1, a_batch).unsqueeze(1)
#         x2_max = global_max_pool(x2, b_batch).unsqueeze(1)

#         # 输入adapters的向量来自GNN输出的向量的mean、max池化的concat
#         x_query = torch.cat([
#             x2_mean,    # [bsz, 1, d]
#             x2_max,    # [bsz, 1, d]
#             x1_mean,   # [bsz, 1, d]
#             x1_max   # [bsz, 1, d]
#         ], dim=1) 


#         # 计算两张图的 cosine similarity
#         x1 = torch.cat([x1_mean, x1_max], dim=1)
#         x2 = torch.cat([x2_mean, x1_max], dim=1)
#         cos_sim = F.cosine_similarity(x1, x2, dim=1)

#         logger.info("===========GNN similarity============")
#         logger.info(cos_sim)
#         logger.info("===========GNN similarity============")

#         x_key = x_query
#         x_value = x_query
        
#         x_query = self.fc_query_adapter(x_query)


#         # 计算门控值
#         gate_val = self.gate(x_key) # gate是通过gnn输出的vector的mean_pool和max_pool调节的？

#         # 分别经过各自的经过adapters
#         x_key = self.fc_key(x_key)
#         x_value = self.fc_value(x_value)

#         # 使用门控进行控制
#         x_key = x_key * gate_val
#         x_value = x_value * gate_val
#         x_kb_queries = x_query * gate_val 

#         # 搬到目标设备
#         x_query = x_query.to(self.target_device)
#         x_kb_queries = x_kb_queries.to(self.target_device)
#         x_key = x_key.to(self.target_device)
#         x_value = x_value.to(self.target_device)

#         x_key_graph = None
#         x_value_graph = None

#         return x_query, x_key, x_value, x_key_graph, x_value_graph, x_kb_queries

# 新增依赖
# from transformers import AutoModelForCausalLM, AutoTokenizer

# class My_GNN_ManualParallel(nn.Module):
'''
    def __init__(
        self,
        args,
        model_name_or_path="/home/sqli/res/vul/lqy_temp/pretrained_models/qwen3_0_6b_model",   # 换成你自己的 Qwen 路径
        num_edge_type=2,
        embedding_dims=None,                   # 不再由外部指定，而是自动读取 LLM hidden_size
        gnn_name='gatedgcn',
        gnn_out_dims=896,
        gnn_layers=4,
        dropout=0.2,
        device_map=None,
        target_device="cuda:0"
    ):
        super().__init__()
        # ---------- 设备规划 ----------
        if device_map is None:
            device_map = ["cuda:0", "cuda:1", "cuda:2"]   # 0 给 LLM，1 给 GNN，2 给 head
        self.llm_device   = torch.device(device_map[0])   # 新增
        self.gnn_device   = torch.device(device_map[1])
        self.head_device  = torch.device(device_map[2])
        self.target_device= torch.device(target_device)

        # ---------- 加载 decoder-only LLM ----------
        logger.info(f"Loading LLM from {model_name_or_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.llm_device},
            trust_remote_code=True,
            use_safetensors=True
        )
        self.llm.eval()                       # 只做推理，不需要训练 LLM
        for p in self.llm.parameters():
            p.requires_grad = False

        # 节点编码器和主干 LLM 可以不是同一个尺寸:
        # 例如节点编码器仍然使用 Qwen3-0.6B(1024)，而主干分类/注入的 LLM 使用 Qwen3-8B(4096)。
        encoder_hidden_size = self.llm.config.hidden_size
        target_llm_hidden_size = getattr(args, "llm_hidden_size", encoder_hidden_size)
        self.node_edge_embedding_size = encoder_hidden_size
        self.target_llm_hidden_size = target_llm_hidden_size
        embedding_dims = encoder_hidden_size

        # ---------- 其余模块照搬，仅把维度改成 llm_hidden_size ----------
        self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims).to(self.gnn_device)

        if gnn_name == 'gatedgcn':
            self.gnn = GatedGCN(
                in_channels=embedding_dims,
                hidden_channels=embedding_dims,
                num_layers=gnn_layers,
                out_channels=embedding_dims,
                dropout=dropout,
                jk='cat',
                edge_dims=embedding_dims
            ).to(self.gnn_device)
        else:
            raise NotImplementedError

        self.layer_norm = nn.LayerNorm(gnn_out_dims).to(self.head_device)

        # Query adapter 负责把图分支表示投影到主干 LLM hidden size。
        self.fc_query_adapter = nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, target_llm_hidden_size, dtype=torch.bfloat16),
            nn.LayerNorm(target_llm_hidden_size)
        ).to(self.head_device)

        self.fc_key= nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16),
            nn.LayerNorm(gnn_out_dims)
        ).to(self.head_device)

        self.fc_value = nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16)
        ).to(self.head_device)

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, 1),
            nn.Sigmoid()
        ).to(self.head_device)

    # ---------- 新增：用 LLM 对「一个 batch 的节点」做 embedding ----------
    @torch.no_grad()
    def _llm_embed_nodes(self, input_ids, attention_mask):
        """
        input_ids:       [total_nodes, seq_len]
        attention_mask:  [total_nodes, seq_len]
        return:          [total_nodes, hidden_size]  已 mean-pool
        """
        llm_device = next(self.llm.parameters()).device
        gnn_device = next(self.gnn.parameters()).device

        input_ids       = input_ids.to(llm_device)
        attention_mask  = attention_mask.to(llm_device)

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # outputs.hidden_states[-1]: [total_nodes, seq_len, hidden]
        last_hidden = outputs.hidden_states[-1]

        # mean-pool 忽略 pad
        mask = attention_mask.unsqueeze(-1)          # [total_nodes, seq_len, 1]
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        # pooled: [total_nodes, hidden]
        return pooled.to(gnn_device)            # 直接搬到 GNN 卡

    # ---------- forward 里把原来的 UniXcoder 部分换成上面函数 ----------
    def forward(self, data):
        llm_device = next(self.llm.parameters()).device
        gnn_device = next(self.gnn.parameters()).device
        head_device = next(self.fc_query_adapter.parameters()).device

        # 1. 搬输入
        a_nodes_ids    = data['a_nodes'].to(llm_device)          # 原来是 token_id，现在直接喂 LLM
        a_attn_mask    = data['a_nodes_attention_mask'].to(llm_device)
        a_edges        = data['a_edges'].to(gnn_device)
        a_edges_type   = data['a_edges_type'].to(gnn_device)

        b_nodes_ids    = data['b_nodes'].to(llm_device)
        b_attn_mask    = data['b_nodes_attention_mask'].to(llm_device)
        b_edges        = data['b_edges'].to(gnn_device)
        b_edges_type   = data['b_edges_type'].to(gnn_device)

        a_batch = data['a_batch']
        b_batch = data['b_batch']

        # 2. LLM 编码节点
        a_encoded = self._llm_embed_nodes(a_nodes_ids, a_attn_mask)   # [N_a, hidden]
        b_encoded = self._llm_embed_nodes(b_nodes_ids, b_attn_mask)

        a_edges_type_emb = self.edge_embedding(a_edges_type)
        b_edges_type_emb = self.edge_embedding(b_edges_type)

        # 3. GNN
        x1 = self.gnn(a_encoded, a_edges, edge_attr=a_edges_type_emb)
        x2 = self.gnn(b_encoded, b_edges, edge_attr=b_edges_type_emb)

        # 4. 后续 pool + adapter 与你原来完全一致
        a_batch = a_batch.to(head_device)
        b_batch = b_batch.to(head_device)
        x1 = x1.to(head_device)
        x2 = x2.to(head_device)

        x1_mean = global_mean_pool(x1, a_batch).unsqueeze(1)
        x2_mean = global_mean_pool(x2, b_batch).unsqueeze(1)
        x1_max  = global_max_pool(x1, a_batch).unsqueeze(1)
        x2_max  = global_max_pool(x2, b_batch).unsqueeze(1)

        x_query = torch.cat([x2_mean, x2_max, x1_mean, x1_max], dim=1)
        x_key   = x_query
        x_value = x_query

        # 5. adapters & gate
        adapter_dtype = next(self.fc_query_adapter.parameters()).dtype
        x_query = x_query.to(device=head_device, dtype=adapter_dtype)
        x_key = x_key.to(device=head_device, dtype=adapter_dtype)
        x_value = x_value.to(device=head_device, dtype=adapter_dtype)
        x_query = self.fc_query_adapter(x_query)
        gate_val = self.gate(x_key)
        x_key   = self.fc_key(x_key)   * gate_val
        x_value = self.fc_value(x_value) * gate_val
        x_kb_queries = x_query * gate_val

        # 6. 搬到最终目标设备
        return (x_query.to(self.target_device),
                x_key.to(self.target_device),
                x_value.to(self.target_device),
                None, None,
                x_kb_queries.to(self.target_device))
'''


# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from torch.nn import Linear
# from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
# from torch_geometric.data import Data
# from transformers import RobertaModel, RobertaConfig
# import lightning as L
# from torch_geometric.nn.models.basic_gnn import BasicGNN
# from model.gsa_pool import GSAPool
# from config.logger import setup_logger
# logger = setup_logger()


# class GatedGCNKernel(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dims):
#         super(GatedGCNKernel, self).__init__(aggr='add')
#         self.A = nn.Linear(in_channels, out_channels)
#         self.B = nn.Linear(in_channels, out_channels)
#         self.C = nn.Linear(edge_dims, out_channels)
#         self.D = nn.Linear(in_channels, out_channels)
#         self.E = nn.Linear(in_channels, out_channels)
    
#     def forward(self, x, edge_index, edge_attr):
#         if isinstance(x, torch.Tensor):
#             x = (x, x) # j -> i

#         Ax = self.A(x[1])

#         Dx = self.D(x[1])
        
#         Bx = self.B(x[0]) # j

#         Ex = self.E(x[0]) # j
        
#         Ce = self.C(edge_attr)

#         out = self.propagate(edge_index, B=Bx, D=Dx, E=Ex, edge_attr=Ce)
        
#         out = out + Ax # Ax是At-1*ht-1
#         # out = F.relu(out + Ax)
#         return out

#     def message(self, B_j, D_i, E_j, edge_attr):
#         gate = torch.sigmoid(D_i + E_j + edge_attr)
#         return gate * B_j


# class GatedGCN_EKernel(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dims):
#         super().__init__(aggr='add')
#         self.A = nn.Linear(in_channels, out_channels)
#         self.B = nn.Linear(in_channels, out_channels)
#         self.C = nn.Linear(edge_dims, out_channels)
#         self.D = nn.Linear(in_channels, out_channels)
#         self.E = nn.Linear(in_channels, out_channels)
    
#     def forward(self, x, edge_index, edge_attr):
#         if isinstance(x, torch.Tensor):
#             x = (x, x) # j -> i
#         Ax = self.A(x[1])
#         Dx = self.D(x[1])
        
#         Bx = self.B(x[0])
#         Ex = self.E(x[0])
        
#         Ce = self.C(edge_attr)
        
#         edge_attr = self.edge_updater(edge_index=edge_index, D=Dx, E=Ex, edge_attr=Ce)
#         out = self.propagate(edge_index, B=Bx, edge_attr=edge_attr)
#         out = out + Ax
        
#         return out, edge_attr

#     def message(self, B_j, edge_attr):
#         gate = torch.sigmoid(edge_attr)
#         return gate * B_j
        
#     def edge_update(self, D_i, E_j, edge_attr):
#         return D_i + E_j + edge_attr
    

# class GatedGCN(BasicGNN):
#     supports_edge_weight = False
#     supports_edge_attr = True
#     def init_conv(self, in_channels: int, out_channels: int, edge_dims: int, **kwargs):
#         return GatedGCNKernel(in_channels, out_channels, edge_dims=edge_dims)

# class My_GNN(L.LightningModule):
#     # embedding_dims 大小等于 unixcoder的hidden size
#     def __init__(
#         self,
#         model_name_or_path='microsoft/unixcoder-base',
#         num_edge_type=2,
#         embedding_dims=768,
#         gnn_name='gatedgcn',
#         llm_hidden_size=4096,
#         gnn_out_dims=896,
#         gnn_layers=4,
#         dropout=0.2
#         # lr=1e-4,
#         # milestones=[30, 60]
#     ):
#         super(My_GNN, self).__init__()
        
#         # transformer layer用UniXcoder
#         self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
#         self.node_edge_embedding_size = self.unixcoder_config.hidden_size

#         self.trans_encoder = RobertaModel(self.unixcoder_config)
#         # 每个token的正常的embedding
#         self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0)
        
#         # 总共有两种边
#         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims)

#         # GatedGCN用来处理被Unixcoder向量化后的CFG信息
#         if gnn_name == 'gatedgcn':
#             self.gnn = GatedGCN(
#                 in_channels=embedding_dims, 
#                 hidden_channels=embedding_dims, 
#                 num_layers=gnn_layers,
#                 out_channels=embedding_dims, 
#                 dropout=dropout, 
#                 jk='cat', 
#                 edge_dims=embedding_dims
#             )
#         self.pool = GSAPool(in_channels = embedding_dims)

#         # self.pooling = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))
#         self.layer_norm = nn.LayerNorm(gnn_out_dims)
        
        
#         # 对齐LLM的hidden dim，方便后续做cross attention
#         self.fc_query = nn.Linear(4*embedding_dims, llm_hidden_size)
#         self.query_layernorm = nn.LayerNorm(llm_hidden_size)
#         self.fc_key = nn.Linear(4*embedding_dims, gnn_out_dims)
#         self.key_layernorm = nn.LayerNorm(gnn_out_dims)
#         self.fc_value = nn.Linear(4*embedding_dims, gnn_out_dims)
#         self.value_layernorm = nn.LayerNorm(gnn_out_dims)

#     def forward(self, data): 
#         # 获取数据
#         a_nodes = data['a_nodes']
#         a_nodes_attention_mask = data['a_nodes_attention_mask']
#         a_edges = data['a_edges']
#         a_edges_type = data['a_edges_type']
#         b_nodes = data['b_nodes']
#         b_nodes_attention_mask = data['b_nodes_attention_mask']
#         b_edges = data['b_edges']
#         b_edges_type = data['b_edges_type']
        
#         # 打印批次中的唯一值
#         print("a_batch unique:", torch.unique(data['a_batch']))  # 应输出 [0, 1, ..., batch_size-1]
#         print("b_batch unique:", torch.unique(data['b_batch']))

#         # 对节点进行token嵌入
#         a_nodes = self.token_embedding(a_nodes)
#         # 对节点进行transformer编码
#         a_nodes = self.trans_encoder(inputs_embeds=a_nodes, attention_mask=a_nodes_attention_mask).pooler_output # 尝试换成Qwen3embedding
#         # 对节点进行token嵌入
#         b_nodes = self.token_embedding(b_nodes)
#         # 对节点进行transformer编码
#         b_nodes = self.trans_encoder(inputs_embeds=b_nodes, attention_mask=b_nodes_attention_mask).pooler_output # 把encoder的最后一层向量x经过一层pooler


#         # 对边类型进行嵌入
#         a_edges_type = self.edge_embedding(a_edges_type)
#         b_edges_type = self.edge_embedding(b_edges_type)

#         # 原始gnn 的方法
#         # 把x和edge一起送入GNN model
#         x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         print("x1.shape() after gnn:" + str(x1.shape))
#         # print("x2.shape() after gnn:" + str(x2.shape))

#         x1 = torch.cat([global_mean_pool(x1, data['a_batch']), global_max_pool(x1, data['a_batch'])], dim=1)
#         x2 = torch.cat([global_mean_pool(x2, data['b_batch']), global_max_pool(x2, data['b_batch'])], dim=1)  # [batch_size, 2*hidden_channels]
#         x = torch.cat([x1, x2], dim=1)  # [batch_size, 4*hidden_channels]

#         print("x1.shape() after cat:" + str(x1.shape))
#         # print("x2.shape() after cat:" + str(x2.shape))

#         x = F.dropout(x, p=0.2, training=self.training)
#         # x = nn.functional.normalize(x, p=2, dim=1)
#         x = x.unsqueeze(1)

#         x_key = self.fc_key(x)
#         x_value = self.fc_value(x)
#         x_query = self.fc_query(x)
#         # x = nn.functional.normalize(x, p=2, dim=1)
        
#         x_key = self.key_layernorm(x_key)
#         x_value = self.value_layernorm(x_value)
#         x_query = self.query_layernorm(x_query)
        
#         return x_query, x_key, x_value

#         # 尝试 1：以 batch 中节点数量最少的 a_node+b_node 来截断
#         # 1. 得到 GNN 输出
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)  # [num_nodes_a, dims]
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)  # [num_nodes_b, dims]

#         # a_batch = data['a_batch']  # [num_nodes_a]
#         # b_batch = data['b_batch']  # [num_nodes_b]

#         # bsz = a_batch.max().item() + 1  # 假设 batch 是 0-based

#         # x_list = []

#         # min_node_num = float('inf')

#         # # 2. 第一次遍历，找出每个图节点数并统计最小值
#         # for i in range(bsz):
#         #     num_nodes_i = (a_batch == i).sum().item() + (b_batch == i).sum().item()
#         #     if num_nodes_i < min_node_num:
#         #         min_node_num = num_nodes_i

#         # # 3. 第二次遍历，拼接每个样本的节点，并截断到最小节点数
#         # for i in range(bsz):
#         #     x1_i = x1[a_batch == i]  # [node_numsa_i, dims]
#         #     x2_i = x2[b_batch == i]  # [node_numsb_i, dims]
#         #     x_i = torch.cat([x1_i, x2_i], dim=0)  # [node_numsa_i + node_numsb_i, dims]

#         #     # 截断前 min_node_num 个节点（保持每个 batch 一样长）
#         #     x_i = x_i[:min_node_num]  # [min_node_num, dims]

#         #     x_list.append(x_i)

#         # # 4. 拼成统一 batch tensor
#         # x = torch.stack(x_list, dim=0)  # [bsz, min_node_num, dims]

#         # x = F.dropout(x, p=0.2, training=self.training)
#         # x = self.fc(x)
#         # # x = nn.functional.normalize(x, p=2, dim=1)
#         # x = self.layer_norm(x)
#         # print("x shape" + str(x.shape))


#         # return x 

#         # 尝试 2：topk，然后按最短的截断
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         # a_batch = data['a_batch']  # shape: [num_total_nodes_a]
#         # b_batch = data['b_batch']  # shape: [num_total_nodes_b]

#         # batch_size_a = a_batch.max().item() + 1
#         # batch_size_b = b_batch.max().item() + 1

#         # # 统计每个样本a图的节点数
#         # a_nodes_count = torch.bincount(a_batch, minlength=batch_size_a)  # tensor长度是batch_size_a
#         # # 统计每个样本b图的节点数
#         # b_nodes_count = torch.bincount(b_batch, minlength=batch_size_b)
        
#         # x1_pool, _, _, batch_a_pool, perm_a = self.pool(
#         #     x1, a_edges, edge_attr=a_edges_type, batch=data['a_batch']
#         # )

#         # # 图 b 的池化
#         # x2_pool, _, _, batch_b_pool, perm_b = self.pool(
#         #     x2, b_edges, edge_attr=b_edges_type, batch=data['b_batch']
#         # )

#         # # 记录每个样本选中的节点数
#         # num_nodes_list = []
#         # num_nodes_list.append(50)

#         # all_samples_nodes = []
#         # batch_size = data['a_batch'].max().item() + 1
#         # for i in range(batch_size):
#         #     a_nodes_i = x1_pool[batch_a_pool == i]  # [num_a_i, hidden_dim]
#         #     b_nodes_i = x2_pool[batch_b_pool == i]  # [num_b_i, hidden_dim]

#         #     all_nodes = torch.cat([a_nodes_i, b_nodes_i], dim=0)  # [num_a_i + num_b_i, hidden_dim]
#         #     num_nodes_list.append(all_nodes.size(0))
#         #     all_samples_nodes.append(all_nodes)

#         # # 对齐：找出最小的节点数
#         # min_nodes = min(num_nodes_list)

#         # # 截断对齐
#         # for i in range(batch_size):
#         #     all_samples_nodes[i] = all_samples_nodes[i][:min_nodes]

#         # # 最终拼接：[batch_size, min_nodes, hidden_dim]
#         # x = torch.stack(all_samples_nodes, dim=0)

#         # x = F.dropout(x, p=0.2, training=self.training)

#         # x_key = self.fc_key(x)
#         # x_value = self.fc_value(x)
#         # x_query = self.fc_query(x)

#         # x_key = self.key_layernorm(x_key)
#         # x_value = self.value_layernorm(x_value)
#         # x_query = self.query_layernorm(x_query)

#         # return x_query, x_key, x_value


#         # # x = nn.functional.normalize(x, p=2, dim=1)
        
#         # x_key = self.layer_norm(x_key)
#         # x_value = self.layer_norm(x_value)
        
#         # return (x_key, x_value)

#         #尝试3： 2张图分别池化
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         # x1_mean = global_mean_pool(x1, data['a_batch']).unsqueeze(1)
#         # x2_mean = global_mean_pool(x2, data['b_batch']).unsqueeze(1)
#         # x1_max = global_max_pool(x1, data['a_batch']).unsqueeze(1)
#         # x2_max = global_max_pool(x2, data['b_batch']).unsqueeze(1)

#         # x = torch.cat([
#         #     x1_mean,   # [bsz, 1, d]
#         #     x1_max,   # [bsz, 1, d]
#         #     x2_mean,    # [bsz, 1, d]
#         #     x2_max     # [bsz, 1, d]
#         # ], dim=1) 

#         # x = F.dropout(x, p=0.2, training=self.training)

#         # x_key = self.fc_key(x)
#         # x_value = self.fc_value(x)

#         # # x = nn.functional.normalize(x, p=2, dim=1)
        
#         # x_key = self.key_layernorm(x_key)
#         # x_value = self.value_layernorm(x_value)
        
#         # return (x_key, x_value)

class My_GNN_ManualParallel(nn.Module):
    def __init__(
        self,
        args,
        model_name_or_path='microsoft/unixcoder-base',
        num_edge_type=2,
        embedding_dims=768,
        gnn_name='gatedgcn',
        gnn_out_dims=896,
        gnn_layers=4,
        dropout=0.2,
        device_map=None,   # 新增：指定模块设备分配，如 ["cuda:1","cuda:2","cuda:3"]
        target_device="cuda:0"  # 最终与LLM对接的设备
    ):
        super().__init__()
        # 设备分配默认值（如果没传 device_map）
        if device_map is None:
            # token/transformer -> cuda:1, gnn -> cuda:2, pool+fc -> cuda:3
            device_map = ["cuda:1", "cuda:2", "cuda:3"]
        assert len(device_map) >= 3, "device_map 至少需要三个设备: [embed_dev, gnn_dev, head_dev]"

        self.embed_device = torch.device(device_map[1])
        self.gnn_device = torch.device(device_map[0])
        self.head_device = torch.device(device_map[2])
        self.target_device = torch.device(target_device)

        # transformer layer 用 UniXcoder
        self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
        self.node_edge_embedding_size = self.unixcoder_config.hidden_size

        # 把较重但可以集中的模块放到 embed_device
        self.trans_encoder = RobertaModel(self.unixcoder_config).to(self.gnn_device)
        self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0).to(self.embed_device)
        self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims).to(self.gnn_device)

        # GatedGCN 放到 gnn_device
        if gnn_name == 'gatedgcn':
            # 假设 GatedGCN 构造函数返回一个 nn.Module，可以 .to(device)
            self.gnn = GatedGCN(
                in_channels=embedding_dims,
                hidden_channels=embedding_dims,
                num_layers=gnn_layers,
                out_channels=embedding_dims,
                dropout=dropout,
                jk='cat',
                edge_dims=embedding_dims
            ).to(self.gnn_device)
        else:
            raise NotImplementedError("目前仅支持 gatedgcn")

        self.layer_norm = nn.LayerNorm(gnn_out_dims).to(self.head_device)

        # 对齐 LLM 的 hidden dim, 放 head_device（最后结果再搬到 target_device）
        # self.fc_query = nn.Linear(embedding_dims, llm_hidden_size).to(self.head_device)
        # self.query_layernorm = nn.LayerNorm(llm_hidden_size).to(self.head_device)
        # Key: Linear -> ReLU -> Linear -> LayerNorm
        # self.fc_key = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)
        # self.fc_value = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)

        # Q K V Adapters
        self.fc_query_adapter = nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, args.llm_hidden_size, dtype=torch.bfloat16),
            nn.LayerNorm(args.llm_hidden_size)
        ).to(self.head_device)

        self.fc_key= nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16),
            nn.LayerNorm(gnn_out_dims)
        ).to(self.head_device)

        self.fc_value = nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16)
        ).to(self.head_device)

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(embedding_dims, gnn_out_dims // 16),
            nn.ReLU(),
            nn.Linear(gnn_out_dims // 16, 1),
            nn.Sigmoid()
        ).to(self.head_device)


        # self.q_adapter_from_kb = nn.Linear(llm_hidden_size, llm_hidden_size, bias=False)

    def to_device(self, obj, device):
        """递归地把 obj（tensor/dict/list/tuple）搬到 device 上"""
        if isinstance(obj, dict):
            return {k: self.to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # 保持原类型
            return type(obj)(self.to_device(v, device) for v in obj)
        elif hasattr(obj, "to"):
            return obj.to(device)
        else:
            return obj

    
    def node_matching(self, x1, x2, threshold=0.80):
        if x1.numel() == 0 or x2.numel() == 0:
            return [], list(range(x1.size(0))), list(range(x2.size(0)))

        sim_matrix = F.cosine_similarity(
            x1.unsqueeze(1),  # [N1, 1, d]
            x2.unsqueeze(0),  # [1, N2, d]
            dim=-1
        )  # [N1, N2]

        cost_matrix = 1 - sim_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches, diffs_pre, diffs_post = [], [], []
        for i, j in zip(row_ind, col_ind):
            sim = sim_matrix[i, j].item()
            if sim >= threshold:
                matches.append((i, j, sim))
            else:
                diffs_pre.append(i)
                diffs_post.append(j)

        matched_pre = {i for i, _, _ in matches}
        matched_post = {j for _, j, _ in matches}

        diffs_pre.extend([i for i in range(x1.size(0)) if i not in matched_pre])
        diffs_post.extend([j for j in range(x2.size(0)) if j not in matched_post])

        return matches, diffs_pre, diffs_post

    def forward(self, data):
        # ---------------------------
        # 1) 把需要的输入先搬到 embed_device
        # ---------------------------
        # data 的字段可能是 tensor 或者其他容器，我们单独把需要的 tensors 搬过去
        a_nodes = data['a_nodes'].to(self.embed_device)
        a_nodes_attention_mask = data['a_nodes_attention_mask'].to(self.gnn_device)
        a_edges = data['a_edges']  # edge index 通常是 LongTensor, 视实现而定
        a_edges_type = data['a_edges_type'].to(self.gnn_device)

        b_nodes = data['b_nodes'].to(self.embed_device)
        b_nodes_attention_mask = data['b_nodes_attention_mask'].to(self.gnn_device)
        b_edges = data['b_edges']
        b_edges_type = data['b_edges_type'].to(self.gnn_device)

        # 批次索引也可能在 cpu，需要明确搬到相应设备用于 pool（pool 会在 head_device 上执行）
        a_batch = data['a_batch']  # 这是 batch idx，用于 pooling，最终会被搬到 head_device
        b_batch = data['b_batch']

        # ---------------------------
        # 2) embedding + transformer 在 embed_device 上计算
        # ---------------------------
        a_nodes_emb = self.token_embedding(a_nodes)   # already on embed_device
        a_encoded = self.trans_encoder(inputs_embeds=a_nodes_emb.to(self.gnn_device), attention_mask=a_nodes_attention_mask).pooler_output
        # a_encoded: [N_a_tokens(number of a graph's nodes?), hidden] or [batch_size_nodes, hidden] 取决于你的输入结构

        b_nodes_emb = self.token_embedding(b_nodes)
        b_encoded = self.trans_encoder(inputs_embeds=b_nodes_emb.to(self.gnn_device), attention_mask=b_nodes_attention_mask).pooler_output

        a_edges_type_emb = self.edge_embedding(a_edges_type)
        b_edges_type_emb = self.edge_embedding(b_edges_type)

        # ---------------------------
        # 3) 把需要给 GNN 的张量搬到 gnn_device
        #    注意 edge_index (a_edges/b_edges) 也要搬
        # ---------------------------
        a_encoded = a_encoded.to(self.gnn_device)
        b_encoded = b_encoded.to(self.gnn_device)
        a_edges = a_edges.to(self.gnn_device)
        b_edges = b_edges.to(self.gnn_device)
        a_edges_type_emb = a_edges_type_emb.to(self.gnn_device)
        b_edges_type_emb = b_edges_type_emb.to(self.gnn_device)

        # ---------------------------
        # 4) 在 gnn_device 上运行 GNN（大部分计算发生在这里）
        # ---------------------------
        x1 = self.gnn(a_encoded, a_edges, edge_attr=a_edges_type_emb)
        x2 = self.gnn(b_encoded, b_edges, edge_attr=b_edges_type_emb)

        a_batch = a_batch.to(self.head_device)
        b_batch = b_batch.to(self.head_device)

        x1 = x1.to(self.head_device)
        x2 = x2.to(self.head_device)
        
        x1_mean = global_mean_pool(x1, a_batch).unsqueeze(1)
        x2_mean = global_mean_pool(x2, b_batch).unsqueeze(1)
        x1_max = global_max_pool(x1, a_batch).unsqueeze(1)
        x2_max = global_max_pool(x2, b_batch).unsqueeze(1)

        # 输入adapters的向量来自GNN输出的向量的mean、max池化的concat
        x_query = torch.cat([
            x2_mean,    # [bsz, 1, d]
            x2_max,    # [bsz, 1, d]
            x1_mean,   # [bsz, 1, d]
            x1_max   # [bsz, 1, d]
        ], dim=1) 
        x_key = x_query
        x_value = x_query
        
        x_query = self.fc_query_adapter(x_query)


        # 计算门控值
        gate_val = self.gate(x_key) # gate是通过gnn输出的vector的mean_pool和max_pool调节的？

        # 分别经过各自的经过adapters
        x_key = self.fc_key(x_key)
        x_value = self.fc_value(x_value)

        # 使用门控进行控制
        x_key = x_key * gate_val
        x_value = x_value * gate_val
        x_kb_queries = x_query * gate_val 

        # 搬到目标设备
        x_query = x_query.to(self.target_device)
        x_kb_queries = x_kb_queries.to(self.target_device)
        x_key = x_key.to(self.target_device)
        x_value = x_value.to(self.target_device)

        x_key_graph = None
        x_value_graph = None

        return x_query, x_key, x_value, x_key_graph, x_value_graph, x_kb_queries

# old 

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from torch.nn import Linear
# from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
# from torch_geometric.data import Data
# from transformers import RobertaModel, RobertaConfig
# import lightning as L
# from torch_geometric.nn.models.basic_gnn import BasicGNN
# from model.gsa_pool import GSAPool
# from config.logger import setup_logger
# logger = setup_logger()


# class GatedGCNKernel(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dims):
#         super(GatedGCNKernel, self).__init__(aggr='add')
#         self.A = nn.Linear(in_channels, out_channels)
#         self.B = nn.Linear(in_channels, out_channels)
#         self.C = nn.Linear(edge_dims, out_channels)
#         self.D = nn.Linear(in_channels, out_channels)
#         self.E = nn.Linear(in_channels, out_channels)
    
#     def forward(self, x, edge_index, edge_attr):
#         if isinstance(x, torch.Tensor):
#             x = (x, x) # j -> i

#         Ax = self.A(x[1])

#         Dx = self.D(x[1])
        
#         Bx = self.B(x[0]) # j

#         Ex = self.E(x[0]) # j
        
#         Ce = self.C(edge_attr)

#         out = self.propagate(edge_index, B=Bx, D=Dx, E=Ex, edge_attr=Ce)
        
#         out = out + Ax # Ax是At-1*ht-1
#         # out = F.relu(out + Ax)
#         return out

#     def message(self, B_j, D_i, E_j, edge_attr):
#         gate = torch.sigmoid(D_i + E_j + edge_attr)
#         return gate * B_j


# class GatedGCN_EKernel(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dims):
#         super().__init__(aggr='add')
#         self.A = nn.Linear(in_channels, out_channels)
#         self.B = nn.Linear(in_channels, out_channels)
#         self.C = nn.Linear(edge_dims, out_channels)
#         self.D = nn.Linear(in_channels, out_channels)
#         self.E = nn.Linear(in_channels, out_channels)
    
#     def forward(self, x, edge_index, edge_attr):
#         if isinstance(x, torch.Tensor):
#             x = (x, x) # j -> i
#         Ax = self.A(x[1])
#         Dx = self.D(x[1])
        
#         Bx = self.B(x[0])
#         Ex = self.E(x[0])
        
#         Ce = self.C(edge_attr)
        
#         edge_attr = self.edge_updater(edge_index=edge_index, D=Dx, E=Ex, edge_attr=Ce)
#         out = self.propagate(edge_index, B=Bx, edge_attr=edge_attr)
#         out = out + Ax
        
#         return out, edge_attr

#     def message(self, B_j, edge_attr):
#         gate = torch.sigmoid(edge_attr)
#         return gate * B_j
        
#     def edge_update(self, D_i, E_j, edge_attr):
#         return D_i + E_j + edge_attr
    

# class GatedGCN(BasicGNN):
#     supports_edge_weight = False
#     supports_edge_attr = True
#     def init_conv(self, in_channels: int, out_channels: int, edge_dims: int, **kwargs):
#         return GatedGCNKernel(in_channels, out_channels, edge_dims=edge_dims)

# class My_GNN(L.LightningModule):
#     # embedding_dims 大小等于 unixcoder的hidden size
#     def __init__(
#         self,
#         model_name_or_path='microsoft/unixcoder-base',
#         num_edge_type=2,
#         embedding_dims=768,
#         gnn_name='gatedgcn',
#         llm_hidden_size=4096,
#         gnn_out_dims=896,
#         gnn_layers=4,
#         dropout=0.2
#         # lr=1e-4,
#         # milestones=[30, 60]
#     ):
#         super(My_GNN, self).__init__()
        
#         # transformer layer用UniXcoder
#         self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
#         self.node_edge_embedding_size = self.unixcoder_config.hidden_size

#         self.trans_encoder = RobertaModel(self.unixcoder_config)
#         # 每个token的正常的embedding
#         self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0)
        
#         # 总共有两种边
#         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims)

#         # GatedGCN用来处理被Unixcoder向量化后的CFG信息
#         if gnn_name == 'gatedgcn':
#             self.gnn = GatedGCN(
#                 in_channels=embedding_dims, 
#                 hidden_channels=embedding_dims, 
#                 num_layers=gnn_layers,
#                 out_channels=embedding_dims, 
#                 dropout=dropout, 
#                 jk='cat', 
#                 edge_dims=embedding_dims
#             )
#         self.pool = GSAPool(in_channels = embedding_dims)

#         # self.pooling = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))
#         self.layer_norm = nn.LayerNorm(gnn_out_dims)
        
        
#         # 对齐LLM的hidden dim，方便后续做cross attention
#         self.fc_query = nn.Linear(4*embedding_dims, llm_hidden_size)
#         self.query_layernorm = nn.LayerNorm(llm_hidden_size)
#         self.fc_key = nn.Linear(4*embedding_dims, gnn_out_dims)
#         self.key_layernorm = nn.LayerNorm(gnn_out_dims)
#         self.fc_value = nn.Linear(4*embedding_dims, gnn_out_dims)
#         self.value_layernorm = nn.LayerNorm(gnn_out_dims)

#     def forward(self, data): 
#         # 获取数据
#         a_nodes = data['a_nodes']
#         a_nodes_attention_mask = data['a_nodes_attention_mask']
#         a_edges = data['a_edges']
#         a_edges_type = data['a_edges_type']
#         b_nodes = data['b_nodes']
#         b_nodes_attention_mask = data['b_nodes_attention_mask']
#         b_edges = data['b_edges']
#         b_edges_type = data['b_edges_type']
        
#         # 打印批次中的唯一值
#         print("a_batch unique:", torch.unique(data['a_batch']))  # 应输出 [0, 1, ..., batch_size-1]
#         print("b_batch unique:", torch.unique(data['b_batch']))

#         # 对节点进行token嵌入
#         a_nodes = self.token_embedding(a_nodes)
#         # 对节点进行transformer编码
#         a_nodes = self.trans_encoder(inputs_embeds=a_nodes, attention_mask=a_nodes_attention_mask).pooler_output # 尝试换成Qwen3embedding
#         # 对节点进行token嵌入
#         b_nodes = self.token_embedding(b_nodes)
#         # 对节点进行transformer编码
#         b_nodes = self.trans_encoder(inputs_embeds=b_nodes, attention_mask=b_nodes_attention_mask).pooler_output # 把encoder的最后一层向量x经过一层pooler


#         # 对边类型进行嵌入
#         a_edges_type = self.edge_embedding(a_edges_type)
#         b_edges_type = self.edge_embedding(b_edges_type)

#         # 原始gnn 的方法
#         # 把x和edge一起送入GNN model
#         x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         print("x1.shape() after gnn:" + str(x1.shape))
#         # print("x2.shape() after gnn:" + str(x2.shape))

#         x1 = torch.cat([global_mean_pool(x1, data['a_batch']), global_max_pool(x1, data['a_batch'])], dim=1)
#         x2 = torch.cat([global_mean_pool(x2, data['b_batch']), global_max_pool(x2, data['b_batch'])], dim=1)  # [batch_size, 2*hidden_channels]
#         x = torch.cat([x1, x2], dim=1)  # [batch_size, 4*hidden_channels]

#         print("x1.shape() after cat:" + str(x1.shape))
#         # print("x2.shape() after cat:" + str(x2.shape))

#         x = F.dropout(x, p=0.2, training=self.training)
#         # x = nn.functional.normalize(x, p=2, dim=1)
#         x = x.unsqueeze(1)

#         x_key = self.fc_key(x)
#         x_value = self.fc_value(x)
#         x_query = self.fc_query(x)
#         # x = nn.functional.normalize(x, p=2, dim=1)
        
#         x_key = self.key_layernorm(x_key)
#         x_value = self.value_layernorm(x_value)
#         x_query = self.query_layernorm(x_query)
        
#         return x_query, x_key, x_value

#         # 尝试 1：以 batch 中节点数量最少的 a_node+b_node 来截断
#         # 1. 得到 GNN 输出
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)  # [num_nodes_a, dims]
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)  # [num_nodes_b, dims]

#         # a_batch = data['a_batch']  # [num_nodes_a]
#         # b_batch = data['b_batch']  # [num_nodes_b]

#         # bsz = a_batch.max().item() + 1  # 假设 batch 是 0-based

#         # x_list = []

#         # min_node_num = float('inf')

#         # # 2. 第一次遍历，找出每个图节点数并统计最小值
#         # for i in range(bsz):
#         #     num_nodes_i = (a_batch == i).sum().item() + (b_batch == i).sum().item()
#         #     if num_nodes_i < min_node_num:
#         #         min_node_num = num_nodes_i

#         # # 3. 第二次遍历，拼接每个样本的节点，并截断到最小节点数
#         # for i in range(bsz):
#         #     x1_i = x1[a_batch == i]  # [node_numsa_i, dims]
#         #     x2_i = x2[b_batch == i]  # [node_numsb_i, dims]
#         #     x_i = torch.cat([x1_i, x2_i], dim=0)  # [node_numsa_i + node_numsb_i, dims]

#         #     # 截断前 min_node_num 个节点（保持每个 batch 一样长）
#         #     x_i = x_i[:min_node_num]  # [min_node_num, dims]

#         #     x_list.append(x_i)

#         # # 4. 拼成统一 batch tensor
#         # x = torch.stack(x_list, dim=0)  # [bsz, min_node_num, dims]

#         # x = F.dropout(x, p=0.2, training=self.training)
#         # x = self.fc(x)
#         # # x = nn.functional.normalize(x, p=2, dim=1)
#         # x = self.layer_norm(x)
#         # print("x shape" + str(x.shape))


#         # return x 

#         # 尝试 2：topk，然后按最短的截断
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         # a_batch = data['a_batch']  # shape: [num_total_nodes_a]
#         # b_batch = data['b_batch']  # shape: [num_total_nodes_b]

#         # batch_size_a = a_batch.max().item() + 1
#         # batch_size_b = b_batch.max().item() + 1

#         # # 统计每个样本a图的节点数
#         # a_nodes_count = torch.bincount(a_batch, minlength=batch_size_a)  # tensor长度是batch_size_a
#         # # 统计每个样本b图的节点数
#         # b_nodes_count = torch.bincount(b_batch, minlength=batch_size_b)
        
#         # x1_pool, _, _, batch_a_pool, perm_a = self.pool(
#         #     x1, a_edges, edge_attr=a_edges_type, batch=data['a_batch']
#         # )

#         # # 图 b 的池化
#         # x2_pool, _, _, batch_b_pool, perm_b = self.pool(
#         #     x2, b_edges, edge_attr=b_edges_type, batch=data['b_batch']
#         # )

#         # # 记录每个样本选中的节点数
#         # num_nodes_list = []
#         # num_nodes_list.append(50)

#         # all_samples_nodes = []
#         # batch_size = data['a_batch'].max().item() + 1
#         # for i in range(batch_size):
#         #     a_nodes_i = x1_pool[batch_a_pool == i]  # [num_a_i, hidden_dim]
#         #     b_nodes_i = x2_pool[batch_b_pool == i]  # [num_b_i, hidden_dim]

#         #     all_nodes = torch.cat([a_nodes_i, b_nodes_i], dim=0)  # [num_a_i + num_b_i, hidden_dim]
#         #     num_nodes_list.append(all_nodes.size(0))
#         #     all_samples_nodes.append(all_nodes)

#         # # 对齐：找出最小的节点数
#         # min_nodes = min(num_nodes_list)

#         # # 截断对齐
#         # for i in range(batch_size):
#         #     all_samples_nodes[i] = all_samples_nodes[i][:min_nodes]

#         # # 最终拼接：[batch_size, min_nodes, hidden_dim]
#         # x = torch.stack(all_samples_nodes, dim=0)

#         # x = F.dropout(x, p=0.2, training=self.training)

#         # x_key = self.fc_key(x)
#         # x_value = self.fc_value(x)
#         # x_query = self.fc_query(x)

#         # x_key = self.key_layernorm(x_key)
#         # x_value = self.value_layernorm(x_value)
#         # x_query = self.query_layernorm(x_query)

#         # return x_query, x_key, x_value


#         # # x = nn.functional.normalize(x, p=2, dim=1)
        
#         # x_key = self.layer_norm(x_key)
#         # x_value = self.layer_norm(x_value)
        
#         # return (x_key, x_value)

#         #尝试3： 2张图分别池化
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         # x1_mean = global_mean_pool(x1, data['a_batch']).unsqueeze(1)
#         # x2_mean = global_mean_pool(x2, data['b_batch']).unsqueeze(1)
#         # x1_max = global_max_pool(x1, data['a_batch']).unsqueeze(1)
#         # x2_max = global_max_pool(x2, data['b_batch']).unsqueeze(1)

#         # x = torch.cat([
#         #     x1_mean,   # [bsz, 1, d]
#         #     x1_max,   # [bsz, 1, d]
#         #     x2_mean,    # [bsz, 1, d]
#         #     x2_max     # [bsz, 1, d]
#         # ], dim=1) 

#         # x = F.dropout(x, p=0.2, training=self.training)

#         # x_key = self.fc_key(x)
#         # x_value = self.fc_value(x)

#         # # x = nn.functional.normalize(x, p=2, dim=1)
        
#         # x_key = self.key_layernorm(x_key)
#         # x_value = self.value_layernorm(x_value)
        
#         # return (x_key, x_value)

# # class My_GNN_ManualParallel(nn.Module):
# #     def __init__(
# #         self,
# #         args,
# #         model_name_or_path='microsoft/unixcoder-base',
# #         num_edge_type=2,
# #         embedding_dims=768,
# #         gnn_name='gatedgcn',
# #         gnn_out_dims=896,
# #         gnn_layers=4,
# #         dropout=0.2,
# #         device_map=None,   # 新增：指定模块设备分配，如 ["cuda:1","cuda:2","cuda:3"]
# #         target_device="cuda:0"  # 最终与LLM对接的设备
# #     ):
# #         super().__init__()
# #         # 设备分配默认值（如果没传 device_map）
# #         if device_map is None:
# #             # token/transformer -> cuda:1, gnn -> cuda:2, pool+fc -> cuda:3
# #             device_map = ["cuda:1", "cuda:2", "cuda:3"]
# #         assert len(device_map) >= 3, "device_map 至少需要三个设备: [embed_dev, gnn_dev, head_dev]"

# #         self.embed_device = torch.device(device_map[1])
# #         self.gnn_device = torch.device(device_map[0])
# #         self.head_device = torch.device(device_map[2])
# #         self.target_device = torch.device(target_device)

# #         # transformer layer 用 UniXcoder
# #         self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
# #         self.node_edge_embedding_size = self.unixcoder_config.hidden_size

# #         # 把较重但可以集中的模块放到 embed_device
# #         self.trans_encoder = RobertaModel(self.unixcoder_config).to(self.gnn_device)
# #         self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0).to(self.embed_device)
# #         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims).to(self.gnn_device)

# #         # GatedGCN 放到 gnn_device
# #         if gnn_name == 'gatedgcn':
# #             # 假设 GatedGCN 构造函数返回一个 nn.Module，可以 .to(device)
# #             self.gnn = GatedGCN(
# #                 in_channels=embedding_dims,
# #                 hidden_channels=embedding_dims,
# #                 num_layers=gnn_layers,
# #                 out_channels=embedding_dims,
# #                 dropout=dropout,
# #                 jk='cat',
# #                 edge_dims=embedding_dims
# #             ).to(self.gnn_device)
# #         else:
# #             raise NotImplementedError("目前仅支持 gatedgcn")

# #         self.layer_norm = nn.LayerNorm(gnn_out_dims).to(self.head_device)

# #         # 对齐 LLM 的 hidden dim, 放 head_device（最后结果再搬到 target_device）
# #         # self.fc_query = nn.Linear(embedding_dims, llm_hidden_size).to(self.head_device)
# #         # self.query_layernorm = nn.LayerNorm(llm_hidden_size).to(self.head_device)
# #         # Key: Linear -> ReLU -> Linear -> LayerNorm
# #         # self.fc_key = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)
# #         # self.fc_value = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)

# #         # Q K V Adapters
# #         self.fc_query_adapter = nn.Sequential(
# #             nn.Linear(embedding_dims, gnn_out_dims // 32, dtype=torch.bfloat16),
# #             nn.ReLU(),
# #             nn.Linear(gnn_out_dims // 32, args.llm_hidden_size, dtype=torch.bfloat16),
# #             nn.LayerNorm(args.llm_hidden_size)
# #         ).to(self.head_device)

# #         self.fc_key= nn.Sequential(
# #             nn.Linear(embedding_dims, gnn_out_dims // 32, dtype=torch.bfloat16),
# #             nn.ReLU(),
# #             nn.Linear(gnn_out_dims // 32, gnn_out_dims, dtype=torch.bfloat16),
# #             nn.LayerNorm(gnn_out_dims)
# #         ).to(self.head_device)

# #         self.fc_value = nn.Sequential(
# #             nn.Linear(embedding_dims, gnn_out_dims // 32, dtype=torch.bfloat16),
# #             nn.ReLU(),
# #             nn.Linear(gnn_out_dims // 32, gnn_out_dims, dtype=torch.bfloat16)
# #         ).to(self.head_device)

# #         # Gate
# #         self.gate = nn.Sequential(
# #             nn.Linear(embedding_dims, gnn_out_dims // 32),
# #             nn.ReLU(),
# #             nn.Linear(gnn_out_dims // 32, 1),
# #             nn.Sigmoid()
# #         ).to(self.head_device)


# #         # self.q_adapter_from_kb = nn.Linear(llm_hidden_size, llm_hidden_size, bias=False)

# #     def to_device(self, obj, device):
# #         """递归地把 obj（tensor/dict/list/tuple）搬到 device 上"""
# #         if isinstance(obj, dict):
# #             return {k: self.to_device(v, device) for k, v in obj.items()}
# #         elif isinstance(obj, (list, tuple)):
# #             # 保持原类型
# #             return type(obj)(self.to_device(v, device) for v in obj)
# #         elif hasattr(obj, "to"):
# #             return obj.to(device)
# #         else:
# #             return obj

    
# #     def node_matching(self, x1, x2, threshold=0.80):
# #         if x1.numel() == 0 or x2.numel() == 0:
# #             return [], list(range(x1.size(0))), list(range(x2.size(0)))

# #         sim_matrix = F.cosine_similarity(
# #             x1.unsqueeze(1),  # [N1, 1, d]
# #             x2.unsqueeze(0),  # [1, N2, d]
# #             dim=-1
# #         )  # [N1, N2]

# #         cost_matrix = 1 - sim_matrix.detach().cpu().numpy()
# #         row_ind, col_ind = linear_sum_assignment(cost_matrix)

# #         matches, diffs_pre, diffs_post = [], [], []
# #         for i, j in zip(row_ind, col_ind):
# #             sim = sim_matrix[i, j].item()
# #             if sim >= threshold:
# #                 matches.append((i, j, sim))
# #             else:
# #                 diffs_pre.append(i)
# #                 diffs_post.append(j)

# #         matched_pre = {i for i, _, _ in matches}
# #         matched_post = {j for _, j, _ in matches}

# #         diffs_pre.extend([i for i in range(x1.size(0)) if i not in matched_pre])
# #         diffs_post.extend([j for j in range(x2.size(0)) if j not in matched_post])

# #         return matches, diffs_pre, diffs_post

# #     def forward(self, data):
# #         # ---------------------------
# #         # 1) 把需要的输入先搬到 embed_device
# #         # ---------------------------
# #         # data 的字段可能是 tensor 或者其他容器，我们单独把需要的 tensors 搬过去
# #         a_nodes = data['a_nodes'].to(self.embed_device)
# #         a_nodes_attention_mask = data['a_nodes_attention_mask'].to(self.gnn_device)
# #         a_edges = data['a_edges']  # edge index 通常是 LongTensor, 视实现而定
# #         a_edges_type = data['a_edges_type'].to(self.gnn_device)

# #         b_nodes = data['b_nodes'].to(self.embed_device)
# #         b_nodes_attention_mask = data['b_nodes_attention_mask'].to(self.gnn_device)
# #         b_edges = data['b_edges']
# #         b_edges_type = data['b_edges_type'].to(self.gnn_device)

# #         # 批次索引也可能在 cpu，需要明确搬到相应设备用于 pool（pool 会在 head_device 上执行）
# #         a_batch = data['a_batch']  # 这是 batch idx，用于 pooling，最终会被搬到 head_device
# #         b_batch = data['b_batch']

# #         # ---------------------------
# #         # 2) embedding + transformer 在 embed_device 上计算
# #         # ---------------------------
# #         a_nodes_emb = self.token_embedding(a_nodes)   # already on embed_device
# #         a_encoded = self.trans_encoder(inputs_embeds=a_nodes_emb.to(self.gnn_device), attention_mask=a_nodes_attention_mask).pooler_output
# #         # a_encoded: [N_a_tokens(number of a graph's nodes?), hidden] or [batch_size_nodes, hidden] 取决于你的输入结构

# #         b_nodes_emb = self.token_embedding(b_nodes)
# #         b_encoded = self.trans_encoder(inputs_embeds=b_nodes_emb.to(self.gnn_device), attention_mask=b_nodes_attention_mask).pooler_output

# #         a_edges_type_emb = self.edge_embedding(a_edges_type)
# #         b_edges_type_emb = self.edge_embedding(b_edges_type)

# #         # ---------------------------
# #         # 3) 把需要给 GNN 的张量搬到 gnn_device
# #         #    注意 edge_index (a_edges/b_edges) 也要搬
# #         # ---------------------------
# #         a_encoded = a_encoded.to(self.gnn_device)
# #         b_encoded = b_encoded.to(self.gnn_device)
# #         a_edges = a_edges.to(self.gnn_device)
# #         b_edges = b_edges.to(self.gnn_device)
# #         a_edges_type_emb = a_edges_type_emb.to(self.gnn_device)
# #         b_edges_type_emb = b_edges_type_emb.to(self.gnn_device)

# #         # ---------------------------
# #         # 4) 在 gnn_device 上运行 GNN（大部分计算发生在这里）
# #         # ---------------------------
# #         x1 = self.gnn(a_encoded, a_edges, edge_attr=a_edges_type_emb)
# #         x2 = self.gnn(b_encoded, b_edges, edge_attr=b_edges_type_emb)

# #         a_batch = a_batch.to(self.head_device)
# #         b_batch = b_batch.to(self.head_device)

# #         x1 = x1.to(self.head_device)
# #         x2 = x2.to(self.head_device)
        
# #         x1_mean = global_mean_pool(x1, a_batch).unsqueeze(1)
# #         x2_mean = global_mean_pool(x2, b_batch).unsqueeze(1)
# #         x1_max = global_max_pool(x1, a_batch).unsqueeze(1)
# #         x2_max = global_max_pool(x2, b_batch).unsqueeze(1)

# #         # 输入adapters的向量来自GNN输出的向量的mean、max池化的concat
# #         x_query = torch.cat([
# #             x2_mean,    # [bsz, 1, d]
# #             x2_max,    # [bsz, 1, d]
# #             x1_mean,   # [bsz, 1, d]
# #             x1_max   # [bsz, 1, d]
# #         ], dim=1) 


# #         # 计算两张图的 cosine similarity
# #         x1 = torch.cat([x1_mean, x1_max], dim=1)
# #         x2 = torch.cat([x2_mean, x1_max], dim=1)
# #         cos_sim = F.cosine_similarity(x1, x2, dim=1)

# #         logger.info("===========GNN similarity============")
# #         logger.info(cos_sim)
# #         logger.info("===========GNN similarity============")

# #         x_key = x_query
# #         x_value = x_query
        
# #         x_query = self.fc_query_adapter(x_query)


# #         # 计算门控值
# #         gate_val = self.gate(x_key) # gate是通过gnn输出的vector的mean_pool和max_pool调节的？

# #         # 分别经过各自的经过adapters
# #         x_key = self.fc_key(x_key)
# #         x_value = self.fc_value(x_value)

# #         # 使用门控进行控制
# #         x_key = x_key * gate_val
# #         x_value = x_value * gate_val
# #         x_kb_queries = x_query * gate_val 

# #         # 搬到目标设备
# #         x_query = x_query.to(self.target_device)
# #         x_kb_queries = x_kb_queries.to(self.target_device)
# #         x_key = x_key.to(self.target_device)
# #         x_value = x_value.to(self.target_device)

# #         x_key_graph = None
# #         x_value_graph = None

# #         return x_query, x_key, x_value, x_key_graph, x_value_graph, x_kb_queries

# # 新增依赖
# from transformers import AutoModelForCausalLM, AutoTokenizer

# class My_GNN_ManualParallel(nn.Module):
#     def __init__(
#         self,
#         args,
#         model_name_or_path="/home/sqli/res/vul/lqy_temp/pretrained_models/qwen3_0_6b_model",   # 换成你自己的 Qwen 路径
#         num_edge_type=2,
#         embedding_dims=None,                   # 不再由外部指定，而是自动读取 LLM hidden_size
#         gnn_name='gatedgcn',
#         gnn_out_dims=896,
#         gnn_layers=4,
#         dropout=0.2,
#         device_map=None,
#         target_device="cuda:0"
#     ):
#         super().__init__()
#         # ---------- 设备规划 ----------
#         if device_map is None:
#             device_map = ["cuda:0", "cuda:1", "cuda:2"]   # 0 给 LLM，1 给 GNN，2 给 head
#         self.llm_device   = torch.device(device_map[0])   # 新增
#         self.gnn_device   = torch.device(device_map[1])
#         self.head_device  = torch.device(device_map[2])
#         self.target_device= torch.device(target_device)

#         # ---------- 加载 decoder-only LLM ----------
#         logger.info(f"Loading LLM from {model_name_or_path}")
#         self.llm = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path,
#             torch_dtype=torch.bfloat16,
#             device_map={"": self.llm_device},
#             trust_remote_code=True,
#             use_safetensors=True
#         )
#         self.llm.eval()                       # 只做推理，不需要训练 LLM
#         for p in self.llm.parameters():
#             p.requires_grad = False

#         # 自动读出维度
#         llm_hidden_size = self.llm.config.hidden_size
#         self.node_edge_embedding_size = llm_hidden_size   # 后续 GNN 输入维度
#         embedding_dims = llm_hidden_size                  # 兼容旧变量名

#         # ---------- 其余模块照搬，仅把维度改成 llm_hidden_size ----------
#         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims).to(self.gnn_device)

#         if gnn_name == 'gatedgcn':
#             self.gnn = GatedGCN(
#                 in_channels=embedding_dims,
#                 hidden_channels=embedding_dims,
#                 num_layers=gnn_layers,
#                 out_channels=embedding_dims,
#                 dropout=dropout,
#                 jk='cat',
#                 edge_dims=embedding_dims
#             ).to(self.gnn_device)
#         else:
#             raise NotImplementedError

#         self.layer_norm = nn.LayerNorm(gnn_out_dims).to(self.head_device)

#         # QKV Adapters 输入维度全部改成 embedding_dims（即 LLM hidden_size）
#         encoder_hidden_size = self.llm.config.hidden_size
#         target_llm_hidden_size = getattr(args, "llm_hidden_size", encoder_hidden_size)

#         self.fc_query_adapter = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, target_llm_hidden_size, dtype=torch.bfloat16),
#             nn.LayerNorm(target_llm_hidden_size)
#         )

#         self.fc_key= nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16),
#             nn.LayerNorm(gnn_out_dims)
#         ).to(self.head_device)

#         self.fc_value = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16)
#         ).to(self.head_device)

#         # Gate
#         self.gate = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, 1),
#             nn.Sigmoid()
#         ).to(self.head_device)

#     # ---------- 新增：用 LLM 对「一个 batch 的节点」做 embedding ----------
#     @torch.no_grad()
#     def _llm_embed_nodes(self, input_ids, attention_mask):
#         """
#         input_ids:       [total_nodes, seq_len]
#         attention_mask:  [total_nodes, seq_len]
#         return:          [total_nodes, hidden_size]  已 mean-pool
#         """
#         input_ids       = input_ids.to(self.llm_device)
#         attention_mask  = attention_mask.to(self.llm_device)

#         outputs = self.llm(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True
#         )
#         # outputs.hidden_states[-1]: [total_nodes, seq_len, hidden]
#         last_hidden = outputs.hidden_states[-1]

#         # mean-pool 忽略 pad
#         mask = attention_mask.unsqueeze(-1)          # [total_nodes, seq_len, 1]
#         pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
#         # pooled: [total_nodes, hidden]
#         return pooled.to(self.gnn_device)            # 直接搬到 GNN 卡

#     # ---------- forward 里把原来的 UniXcoder 部分换成上面函数 ----------
#     def forward(self, data):
#         # 1. 搬输入
#         a_nodes_ids    = data['a_nodes'].to(self.llm_device)          # 原来是 token_id，现在直接喂 LLM
#         a_attn_mask    = data['a_nodes_attention_mask'].to(self.llm_device)
#         a_edges        = data['a_edges'].to(self.gnn_device)
#         a_edges_type   = data['a_edges_type'].to(self.gnn_device)

#         b_nodes_ids    = data['b_nodes'].to(self.llm_device)
#         b_attn_mask    = data['b_nodes_attention_mask'].to(self.llm_device)
#         b_edges        = data['b_edges'].to(self.gnn_device)
#         b_edges_type   = data['b_edges_type'].to(self.gnn_device)

#         a_batch = data['a_batch']
#         b_batch = data['b_batch']

#         # 2. LLM 编码节点
#         a_encoded = self._llm_embed_nodes(a_nodes_ids, a_attn_mask)   # [N_a, hidden]
#         b_encoded = self._llm_embed_nodes(b_nodes_ids, b_attn_mask)

#         a_edges_type_emb = self.edge_embedding(a_edges_type)
#         b_edges_type_emb = self.edge_embedding(b_edges_type)

#         # 3. GNN
#         x1 = self.gnn(a_encoded, a_edges, edge_attr=a_edges_type_emb)
#         x2 = self.gnn(b_encoded, b_edges, edge_attr=b_edges_type_emb)

#         # 4. 后续 pool + adapter 与你原来完全一致
#         a_batch = a_batch.to(self.head_device)
#         b_batch = b_batch.to(self.head_device)
#         x1 = x1.to(self.head_device)
#         x2 = x2.to(self.head_device)

#         x1_mean = global_mean_pool(x1, a_batch).unsqueeze(1)
#         x2_mean = global_mean_pool(x2, b_batch).unsqueeze(1)
#         x1_max  = global_max_pool(x1, a_batch).unsqueeze(1)
#         x2_max  = global_max_pool(x2, b_batch).unsqueeze(1)

#         x_query = torch.cat([x2_mean, x2_max, x1_mean, x1_max], dim=1)
#         x_key   = x_query
#         x_value = x_query

#         # 5. adapters & gate
#         x_query = self.fc_query_adapter(x_query)
#         gate_val = self.gate(x_key)
#         x_key   = self.fc_key(x_key)   * gate_val
#         x_value = self.fc_value(x_value) * gate_val
#         x_kb_queries = x_query * gate_val

#         # 6. 搬到最终目标设备
#         return (x_query.to(self.target_device),
#                 x_key.to(self.target_device),
#                 x_value.to(self.target_device),
#                 None, None,
#                 x_kb_queries.to(self.target_device))


# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from torch.nn import Linear
# from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
# from torch_geometric.data import Data
# from transformers import RobertaModel, RobertaConfig
# import lightning as L
# from torch_geometric.nn.models.basic_gnn import BasicGNN
# from model.gsa_pool import GSAPool
# from config.logger import setup_logger
# logger = setup_logger()


# class GatedGCNKernel(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dims):
#         super(GatedGCNKernel, self).__init__(aggr='add')
#         self.A = nn.Linear(in_channels, out_channels)
#         self.B = nn.Linear(in_channels, out_channels)
#         self.C = nn.Linear(edge_dims, out_channels)
#         self.D = nn.Linear(in_channels, out_channels)
#         self.E = nn.Linear(in_channels, out_channels)
    
#     def forward(self, x, edge_index, edge_attr):
#         if isinstance(x, torch.Tensor):
#             x = (x, x) # j -> i

#         Ax = self.A(x[1])

#         Dx = self.D(x[1])
        
#         Bx = self.B(x[0]) # j

#         Ex = self.E(x[0]) # j
        
#         Ce = self.C(edge_attr)

#         out = self.propagate(edge_index, B=Bx, D=Dx, E=Ex, edge_attr=Ce)
        
#         out = out + Ax # Ax是At-1*ht-1
#         # out = F.relu(out + Ax)
#         return out

#     def message(self, B_j, D_i, E_j, edge_attr):
#         gate = torch.sigmoid(D_i + E_j + edge_attr)
#         return gate * B_j


# class GatedGCN_EKernel(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dims):
#         super().__init__(aggr='add')
#         self.A = nn.Linear(in_channels, out_channels)
#         self.B = nn.Linear(in_channels, out_channels)
#         self.C = nn.Linear(edge_dims, out_channels)
#         self.D = nn.Linear(in_channels, out_channels)
#         self.E = nn.Linear(in_channels, out_channels)
    
#     def forward(self, x, edge_index, edge_attr):
#         if isinstance(x, torch.Tensor):
#             x = (x, x) # j -> i
#         Ax = self.A(x[1])
#         Dx = self.D(x[1])
        
#         Bx = self.B(x[0])
#         Ex = self.E(x[0])
        
#         Ce = self.C(edge_attr)
        
#         edge_attr = self.edge_updater(edge_index=edge_index, D=Dx, E=Ex, edge_attr=Ce)
#         out = self.propagate(edge_index, B=Bx, edge_attr=edge_attr)
#         out = out + Ax
        
#         return out, edge_attr

#     def message(self, B_j, edge_attr):
#         gate = torch.sigmoid(edge_attr)
#         return gate * B_j
        
#     def edge_update(self, D_i, E_j, edge_attr):
#         return D_i + E_j + edge_attr
    

# class GatedGCN(BasicGNN):
#     supports_edge_weight = False
#     supports_edge_attr = True
#     def init_conv(self, in_channels: int, out_channels: int, edge_dims: int, **kwargs):
#         return GatedGCNKernel(in_channels, out_channels, edge_dims=edge_dims)

# class My_GNN(L.LightningModule):
#     # embedding_dims 大小等于 unixcoder的hidden size
#     def __init__(
#         self,
#         model_name_or_path='microsoft/unixcoder-base',
#         num_edge_type=2,
#         embedding_dims=768,
#         gnn_name='gatedgcn',
#         llm_hidden_size=4096,
#         gnn_out_dims=896,
#         gnn_layers=4,
#         dropout=0.2
#         # lr=1e-4,
#         # milestones=[30, 60]
#     ):
#         super(My_GNN, self).__init__()
        
#         # transformer layer用UniXcoder
#         self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
#         self.node_edge_embedding_size = self.unixcoder_config.hidden_size

#         self.trans_encoder = RobertaModel(self.unixcoder_config)
#         # 每个token的正常的embedding
#         self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0)
        
#         # 总共有两种边
#         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims)

#         # GatedGCN用来处理被Unixcoder向量化后的CFG信息
#         if gnn_name == 'gatedgcn':
#             self.gnn = GatedGCN(
#                 in_channels=embedding_dims, 
#                 hidden_channels=embedding_dims, 
#                 num_layers=gnn_layers,
#                 out_channels=embedding_dims, 
#                 dropout=dropout, 
#                 jk='cat', 
#                 edge_dims=embedding_dims
#             )
#         self.pool = GSAPool(in_channels = embedding_dims)

#         # self.pooling = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))
#         self.layer_norm = nn.LayerNorm(gnn_out_dims)
        
        
#         # 对齐LLM的hidden dim，方便后续做cross attention
#         self.fc_query = nn.Linear(4*embedding_dims, llm_hidden_size)
#         self.query_layernorm = nn.LayerNorm(llm_hidden_size)
#         self.fc_key = nn.Linear(4*embedding_dims, gnn_out_dims)
#         self.key_layernorm = nn.LayerNorm(gnn_out_dims)
#         self.fc_value = nn.Linear(4*embedding_dims, gnn_out_dims)
#         self.value_layernorm = nn.LayerNorm(gnn_out_dims)

#     def forward(self, data): 
#         # 获取数据
#         a_nodes = data['a_nodes']
#         a_nodes_attention_mask = data['a_nodes_attention_mask']
#         a_edges = data['a_edges']
#         a_edges_type = data['a_edges_type']
#         b_nodes = data['b_nodes']
#         b_nodes_attention_mask = data['b_nodes_attention_mask']
#         b_edges = data['b_edges']
#         b_edges_type = data['b_edges_type']
        
#         # 打印批次中的唯一值
#         print("a_batch unique:", torch.unique(data['a_batch']))  # 应输出 [0, 1, ..., batch_size-1]
#         print("b_batch unique:", torch.unique(data['b_batch']))

#         # 对节点进行token嵌入
#         a_nodes = self.token_embedding(a_nodes)
#         # 对节点进行transformer编码
#         a_nodes = self.trans_encoder(inputs_embeds=a_nodes, attention_mask=a_nodes_attention_mask).pooler_output # 尝试换成Qwen3embedding
#         # 对节点进行token嵌入
#         b_nodes = self.token_embedding(b_nodes)
#         # 对节点进行transformer编码
#         b_nodes = self.trans_encoder(inputs_embeds=b_nodes, attention_mask=b_nodes_attention_mask).pooler_output # 把encoder的最后一层向量x经过一层pooler


#         # 对边类型进行嵌入
#         a_edges_type = self.edge_embedding(a_edges_type)
#         b_edges_type = self.edge_embedding(b_edges_type)

#         # 原始gnn 的方法
#         # 把x和edge一起送入GNN model
#         x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         print("x1.shape() after gnn:" + str(x1.shape))
#         # print("x2.shape() after gnn:" + str(x2.shape))

#         x1 = torch.cat([global_mean_pool(x1, data['a_batch']), global_max_pool(x1, data['a_batch'])], dim=1)
#         x2 = torch.cat([global_mean_pool(x2, data['b_batch']), global_max_pool(x2, data['b_batch'])], dim=1)  # [batch_size, 2*hidden_channels]
#         x = torch.cat([x1, x2], dim=1)  # [batch_size, 4*hidden_channels]

#         print("x1.shape() after cat:" + str(x1.shape))
#         # print("x2.shape() after cat:" + str(x2.shape))

#         x = F.dropout(x, p=0.2, training=self.training)
#         # x = nn.functional.normalize(x, p=2, dim=1)
#         x = x.unsqueeze(1)

#         x_key = self.fc_key(x)
#         x_value = self.fc_value(x)
#         x_query = self.fc_query(x)
#         # x = nn.functional.normalize(x, p=2, dim=1)
        
#         x_key = self.key_layernorm(x_key)
#         x_value = self.value_layernorm(x_value)
#         x_query = self.query_layernorm(x_query)
        
#         return x_query, x_key, x_value

#         # 尝试 1：以 batch 中节点数量最少的 a_node+b_node 来截断
#         # 1. 得到 GNN 输出
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)  # [num_nodes_a, dims]
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)  # [num_nodes_b, dims]

#         # a_batch = data['a_batch']  # [num_nodes_a]
#         # b_batch = data['b_batch']  # [num_nodes_b]

#         # bsz = a_batch.max().item() + 1  # 假设 batch 是 0-based

#         # x_list = []

#         # min_node_num = float('inf')

#         # # 2. 第一次遍历，找出每个图节点数并统计最小值
#         # for i in range(bsz):
#         #     num_nodes_i = (a_batch == i).sum().item() + (b_batch == i).sum().item()
#         #     if num_nodes_i < min_node_num:
#         #         min_node_num = num_nodes_i

#         # # 3. 第二次遍历，拼接每个样本的节点，并截断到最小节点数
#         # for i in range(bsz):
#         #     x1_i = x1[a_batch == i]  # [node_numsa_i, dims]
#         #     x2_i = x2[b_batch == i]  # [node_numsb_i, dims]
#         #     x_i = torch.cat([x1_i, x2_i], dim=0)  # [node_numsa_i + node_numsb_i, dims]

#         #     # 截断前 min_node_num 个节点（保持每个 batch 一样长）
#         #     x_i = x_i[:min_node_num]  # [min_node_num, dims]

#         #     x_list.append(x_i)

#         # # 4. 拼成统一 batch tensor
#         # x = torch.stack(x_list, dim=0)  # [bsz, min_node_num, dims]

#         # x = F.dropout(x, p=0.2, training=self.training)
#         # x = self.fc(x)
#         # # x = nn.functional.normalize(x, p=2, dim=1)
#         # x = self.layer_norm(x)
#         # print("x shape" + str(x.shape))


#         # return x 

#         # 尝试 2：topk，然后按最短的截断
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         # a_batch = data['a_batch']  # shape: [num_total_nodes_a]
#         # b_batch = data['b_batch']  # shape: [num_total_nodes_b]

#         # batch_size_a = a_batch.max().item() + 1
#         # batch_size_b = b_batch.max().item() + 1

#         # # 统计每个样本a图的节点数
#         # a_nodes_count = torch.bincount(a_batch, minlength=batch_size_a)  # tensor长度是batch_size_a
#         # # 统计每个样本b图的节点数
#         # b_nodes_count = torch.bincount(b_batch, minlength=batch_size_b)
        
#         # x1_pool, _, _, batch_a_pool, perm_a = self.pool(
#         #     x1, a_edges, edge_attr=a_edges_type, batch=data['a_batch']
#         # )

#         # # 图 b 的池化
#         # x2_pool, _, _, batch_b_pool, perm_b = self.pool(
#         #     x2, b_edges, edge_attr=b_edges_type, batch=data['b_batch']
#         # )

#         # # 记录每个样本选中的节点数
#         # num_nodes_list = []
#         # num_nodes_list.append(50)

#         # all_samples_nodes = []
#         # batch_size = data['a_batch'].max().item() + 1
#         # for i in range(batch_size):
#         #     a_nodes_i = x1_pool[batch_a_pool == i]  # [num_a_i, hidden_dim]
#         #     b_nodes_i = x2_pool[batch_b_pool == i]  # [num_b_i, hidden_dim]

#         #     all_nodes = torch.cat([a_nodes_i, b_nodes_i], dim=0)  # [num_a_i + num_b_i, hidden_dim]
#         #     num_nodes_list.append(all_nodes.size(0))
#         #     all_samples_nodes.append(all_nodes)

#         # # 对齐：找出最小的节点数
#         # min_nodes = min(num_nodes_list)

#         # # 截断对齐
#         # for i in range(batch_size):
#         #     all_samples_nodes[i] = all_samples_nodes[i][:min_nodes]

#         # # 最终拼接：[batch_size, min_nodes, hidden_dim]
#         # x = torch.stack(all_samples_nodes, dim=0)

#         # x = F.dropout(x, p=0.2, training=self.training)

#         # x_key = self.fc_key(x)
#         # x_value = self.fc_value(x)
#         # x_query = self.fc_query(x)

#         # x_key = self.key_layernorm(x_key)
#         # x_value = self.value_layernorm(x_value)
#         # x_query = self.query_layernorm(x_query)

#         # return x_query, x_key, x_value


#         # # x = nn.functional.normalize(x, p=2, dim=1)
        
#         # x_key = self.layer_norm(x_key)
#         # x_value = self.layer_norm(x_value)
        
#         # return (x_key, x_value)

#         #尝试3： 2张图分别池化
#         # x1 = self.gnn(a_nodes, a_edges, edge_attr=a_edges_type)
#         # x2 = self.gnn(b_nodes, b_edges, edge_attr=b_edges_type)

#         # x1_mean = global_mean_pool(x1, data['a_batch']).unsqueeze(1)
#         # x2_mean = global_mean_pool(x2, data['b_batch']).unsqueeze(1)
#         # x1_max = global_max_pool(x1, data['a_batch']).unsqueeze(1)
#         # x2_max = global_max_pool(x2, data['b_batch']).unsqueeze(1)

#         # x = torch.cat([
#         #     x1_mean,   # [bsz, 1, d]
#         #     x1_max,   # [bsz, 1, d]
#         #     x2_mean,    # [bsz, 1, d]
#         #     x2_max     # [bsz, 1, d]
#         # ], dim=1) 

#         # x = F.dropout(x, p=0.2, training=self.training)

#         # x_key = self.fc_key(x)
#         # x_value = self.fc_value(x)

#         # # x = nn.functional.normalize(x, p=2, dim=1)
        
#         # x_key = self.key_layernorm(x_key)
#         # x_value = self.value_layernorm(x_value)
        
#         # return (x_key, x_value)

# class My_GNN_ManualParallel(nn.Module):
#     def __init__(
#         self,
#         model_name_or_path='microsoft/unixcoder-base',
#         num_edge_type=2,
#         embedding_dims=768,
#         gnn_name='gatedgcn',
#         llm_hidden_size=4096,
#         gnn_out_dims=896,
#         gnn_layers=4,
#         dropout=0.2,
#         device_map=None,   # 新增：指定模块设备分配，如 ["cuda:1","cuda:2","cuda:3"]
#         target_device="cuda:0"  # 最终与LLM对接的设备
#     ):
#         super().__init__()
#         # 设备分配默认值（如果没传 device_map）
#         if device_map is None:
#             # token/transformer -> cuda:1, gnn -> cuda:2, pool+fc -> cuda:3
#             device_map = ["cuda:1", "cuda:2", "cuda:3"]
#         assert len(device_map) >= 3, "device_map 至少需要三个设备: [embed_dev, gnn_dev, head_dev]"

#         self.embed_device = torch.device(device_map[1])
#         self.gnn_device = torch.device(device_map[0])
#         self.head_device = torch.device(device_map[2])
#         self.target_device = torch.device(target_device)

#         # transformer layer 用 UniXcoder
#         self.unixcoder_config = RobertaConfig.from_pretrained(model_name_or_path)
#         self.node_edge_embedding_size = self.unixcoder_config.hidden_size

#         # 把较重但可以集中的模块放到 embed_device
#         self.trans_encoder = RobertaModel(self.unixcoder_config).to(self.gnn_device)
#         self.token_embedding = nn.Embedding(self.unixcoder_config.vocab_size, embedding_dims, padding_idx=0).to(self.embed_device)
#         self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims).to(self.gnn_device)

#         # GatedGCN 放到 gnn_device
#         if gnn_name == 'gatedgcn':
#             # 假设 GatedGCN 构造函数返回一个 nn.Module，可以 .to(device)
#             self.gnn = GatedGCN(
#                 in_channels=embedding_dims,
#                 hidden_channels=embedding_dims,
#                 num_layers=gnn_layers,
#                 out_channels=embedding_dims,
#                 dropout=dropout,
#                 jk='cat',
#                 edge_dims=embedding_dims
#             ).to(self.gnn_device)
#         else:
#             raise NotImplementedError("目前仅支持 gatedgcn")

#         self.layer_norm = nn.LayerNorm(gnn_out_dims).to(self.head_device)

#         # 对齐 LLM 的 hidden dim, 放 head_device（最后结果再搬到 target_device）
#         # self.fc_query = nn.Linear(embedding_dims, llm_hidden_size).to(self.head_device)
#         # self.query_layernorm = nn.LayerNorm(llm_hidden_size).to(self.head_device)
#         # Key: Linear -> ReLU -> Linear -> LayerNorm
#         # self.fc_key = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)
#         # self.fc_value = nn.Linear(embedding_dims, gnn_out_dims).to(self.head_device)

#         # Q K V Adapters
#         self.fc_query_adapter = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, llm_hidden_size, dtype=torch.bfloat16),
#             nn.LayerNorm(llm_hidden_size)
#         ).to(self.head_device)

#         self.fc_key= nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16),
#             nn.LayerNorm(gnn_out_dims)
#         ).to(self.head_device)

#         self.fc_value = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16, dtype=torch.bfloat16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, gnn_out_dims, dtype=torch.bfloat16)
#         ).to(self.head_device)

#         # Gate
#         self.gate = nn.Sequential(
#             nn.Linear(embedding_dims, gnn_out_dims // 16),
#             nn.ReLU(),
#             nn.Linear(gnn_out_dims // 16, 1),
#             nn.Sigmoid()
#         ).to(self.head_device)


#         # self.q_adapter_from_kb = nn.Linear(llm_hidden_size, llm_hidden_size, bias=False)

#     def to_device(self, obj, device):
#         """递归地把 obj（tensor/dict/list/tuple）搬到 device 上"""
#         if isinstance(obj, dict):
#             return {k: self.to_device(v, device) for k, v in obj.items()}
#         elif isinstance(obj, (list, tuple)):
#             # 保持原类型
#             return type(obj)(self.to_device(v, device) for v in obj)
#         elif hasattr(obj, "to"):
#             return obj.to(device)
#         else:
#             return obj

    
#     def node_matching(self, x1, x2, threshold=0.80):
#         if x1.numel() == 0 or x2.numel() == 0:
#             return [], list(range(x1.size(0))), list(range(x2.size(0)))

#         sim_matrix = F.cosine_similarity(
#             x1.unsqueeze(1),  # [N1, 1, d]
#             x2.unsqueeze(0),  # [1, N2, d]
#             dim=-1
#         )  # [N1, N2]

#         cost_matrix = 1 - sim_matrix.detach().cpu().numpy()
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         matches, diffs_pre, diffs_post = [], [], []
#         for i, j in zip(row_ind, col_ind):
#             sim = sim_matrix[i, j].item()
#             if sim >= threshold:
#                 matches.append((i, j, sim))
#             else:
#                 diffs_pre.append(i)
#                 diffs_post.append(j)

#         matched_pre = {i for i, _, _ in matches}
#         matched_post = {j for _, j, _ in matches}

#         diffs_pre.extend([i for i in range(x1.size(0)) if i not in matched_pre])
#         diffs_post.extend([j for j in range(x2.size(0)) if j not in matched_post])

#         return matches, diffs_pre, diffs_post

#     def forward(self, data):
#         # ---------------------------
#         # 1) 把需要的输入先搬到 embed_device
#         # ---------------------------
#         # data 的字段可能是 tensor 或者其他容器，我们单独把需要的 tensors 搬过去
#         a_nodes = data['a_nodes'].to(self.embed_device)
#         a_nodes_attention_mask = data['a_nodes_attention_mask'].to(self.gnn_device)
#         a_edges = data['a_edges']  # edge index 通常是 LongTensor, 视实现而定
#         a_edges_type = data['a_edges_type'].to(self.gnn_device)

#         b_nodes = data['b_nodes'].to(self.embed_device)
#         b_nodes_attention_mask = data['b_nodes_attention_mask'].to(self.gnn_device)
#         b_edges = data['b_edges']
#         b_edges_type = data['b_edges_type'].to(self.gnn_device)

#         # 批次索引也可能在 cpu，需要明确搬到相应设备用于 pool（pool 会在 head_device 上执行）
#         a_batch = data['a_batch']  # 这是 batch idx，用于 pooling，最终会被搬到 head_device
#         b_batch = data['b_batch']

#         # ---------------------------
#         # 2) embedding + transformer 在 embed_device 上计算
#         # ---------------------------
#         a_nodes_emb = self.token_embedding(a_nodes)   # already on embed_device
#         a_encoded = self.trans_encoder(inputs_embeds=a_nodes_emb.to(self.gnn_device), attention_mask=a_nodes_attention_mask).pooler_output
#         # a_encoded: [N_a_tokens(number of a graph's nodes?), hidden] or [batch_size_nodes, hidden] 取决于你的输入结构

#         b_nodes_emb = self.token_embedding(b_nodes)
#         b_encoded = self.trans_encoder(inputs_embeds=b_nodes_emb.to(self.gnn_device), attention_mask=b_nodes_attention_mask).pooler_output

#         a_edges_type_emb = self.edge_embedding(a_edges_type)
#         b_edges_type_emb = self.edge_embedding(b_edges_type)

#         # ---------------------------
#         # 3) 把需要给 GNN 的张量搬到 gnn_device
#         #    注意 edge_index (a_edges/b_edges) 也要搬
#         # ---------------------------
#         a_encoded = a_encoded.to(self.gnn_device)
#         b_encoded = b_encoded.to(self.gnn_device)
#         a_edges = a_edges.to(self.gnn_device)
#         b_edges = b_edges.to(self.gnn_device)
#         a_edges_type_emb = a_edges_type_emb.to(self.gnn_device)
#         b_edges_type_emb = b_edges_type_emb.to(self.gnn_device)

#         # ---------------------------
#         # 4) 在 gnn_device 上运行 GNN（大部分计算发生在这里）
#         # ---------------------------
#         x1 = self.gnn(a_encoded, a_edges, edge_attr=a_edges_type_emb)
#         x2 = self.gnn(b_encoded, b_edges, edge_attr=b_edges_type_emb)

#         a_batch = a_batch.to(self.head_device)
#         b_batch = b_batch.to(self.head_device)

#         x1 = x1.to(self.head_device)
#         x2 = x2.to(self.head_device)
        
#         x1_mean = global_mean_pool(x1, a_batch).unsqueeze(1)
#         x2_mean = global_mean_pool(x2, b_batch).unsqueeze(1)
#         x1_max = global_max_pool(x1, a_batch).unsqueeze(1)
#         x2_max = global_max_pool(x2, b_batch).unsqueeze(1)

#         # 输入adapters的向量来自GNN输出的向量的mean、max池化的concat
#         x_query = torch.cat([
#             x2_mean,    # [bsz, 1, d]
#             x2_max,    # [bsz, 1, d]
#             x1_mean,   # [bsz, 1, d]
#             x1_max   # [bsz, 1, d]
#         ], dim=1) 
#         x_key = x_query
#         x_value = x_query
        
#         x_query = self.fc_query_adapter(x_query)


#         # 计算门控值
#         gate_val = self.gate(x_key) # gate是通过gnn输出的vector的mean_pool和max_pool调节的？

#         # 分别经过各自的经过adapters
#         x_key = self.fc_key(x_key)
#         x_value = self.fc_value(x_value)

#         # 使用门控进行控制
#         x_key = x_key * gate_val
#         x_value = x_value * gate_val
#         x_kb_queries = x_query * gate_val 

#         # 搬到目标设备
#         x_query = x_query.to(self.target_device)
#         x_kb_queries = x_kb_queries.to(self.target_device)
#         x_key = x_key.to(self.target_device)
#         x_value = x_value.to(self.target_device)

#         x_key_graph = None
#         x_value_graph = None

#         return x_query, x_key, x_value, x_key_graph, x_value_graph, x_kb_queries
