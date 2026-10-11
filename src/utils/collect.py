import torch
from config.logger import setup_logger
logger = setup_logger()
def custom_collate_fn(batch):
    # 1. 拼接节点特征，直接按第0维拼接
    a_nodes = torch.cat([item.a_nodes for item in batch], dim=0)  # [sum_nodes, 512]
    b_nodes = torch.cat([item.b_nodes for item in batch], dim=0)
    
    a_nodes_attention_mask = torch.cat([item.a_nodes_attention_mask for item in batch], dim=0)
    b_nodes_attention_mask = torch.cat([item.b_nodes_attention_mask for item in batch], dim=0)

    # 2. 拼接边索引和边类型，需节点偏移
    a_edge_list, b_edge_list = [], []
    a_edge_type_list, b_edge_type_list = [], []
    a_offset, b_offset = 0, 0

    for item in batch:
        if item.a_edges.numel() > 0:
            a_edge_list.append(item.a_edges + a_offset)
            a_edge_type_list.append(item.a_edges_type)
        if item.b_edges.numel() > 0:
            b_edge_list.append(item.b_edges + b_offset)
            b_edge_type_list.append(item.b_edges_type)

        a_offset += item.a_nodes.size(0)
        b_offset += item.b_nodes.size(0)

    a_edges = torch.cat(a_edge_list, dim=1) if a_edge_list else torch.empty((2, 0), dtype=torch.long)
    b_edges = torch.cat(b_edge_list, dim=1) if b_edge_list else torch.empty((2, 0), dtype=torch.long)

    a_edges_type = torch.cat(a_edge_type_list, dim=0) if a_edge_type_list else torch.empty((0,), dtype=torch.long)
    b_edges_type = torch.cat(b_edge_type_list, dim=0) if b_edge_type_list else torch.empty((0,), dtype=torch.long)

    # 3. 拼接batch索引
    a_batch = torch.cat([
        torch.full((item.a_nodes.size(0),), i, dtype=torch.long) 
        for i, item in enumerate(batch)
    ], dim=0)
    
    b_batch = torch.cat([
        torch.full((item.b_nodes.size(0),), i, dtype=torch.long) 
        for i, item in enumerate(batch)
    ], dim=0)

    # 4. 处理seq_input字典，逐字段stack
    seq_input_keys = batch[0].seq_input.keys()
    seq_input = {}
    for k in seq_input_keys:
        values = [item.seq_input[k] for item in batch]
        # 如果是tensor，stack；否则保留list
        if isinstance(values[0], torch.Tensor):
            seq_input[k] = torch.cat(values, dim=0)
        else:
            seq_input[k] = values  # 例如字符串等非tensor数据

    # 5. 标签直接stack
    labels = torch.stack([item.label for item in batch], dim=0)
    return {
        'a_nodes': a_nodes,
        'b_nodes': b_nodes,
        'a_nodes_attention_mask': a_nodes_attention_mask,
        'b_nodes_attention_mask': b_nodes_attention_mask,
        'a_edges': a_edges,
        'b_edges': b_edges,
        'a_edges_type': a_edges_type,
        'b_edges_type': b_edges_type,
        'a_batch': a_batch,
        'b_batch': b_batch,
        'seq_input': seq_input,
        'label': labels
    }
