import os
import json,difflib
import torch
from torch_geometric.data import Data
import numpy as np

from config.logger import setup_logger
from utils.helper import RunTime
from data.process_graph import process_graph

logger = setup_logger()

def GetDataset(node_tokenizer, seq_tokenizer, root_path, ass_path, pse_path, mode='train'):
    '''
    Get the dataset from numpy data files.
    :param path: the path used to store numpy dataset.
    :return: dataset - list of torch_geometric.data.Data
    '''

    with open(os.path.join(root_path, mode+r'_infos.json'), 'r') as f:
        data_infos = json.load(f)
    
    inputs_list = []

    count = 0
    for idx, info_item in enumerate(data_infos):
        # a files
        # if ass_path == None or info_item["project_dir"] == None or info_item["optimize_level"] == None or info_item["a_ass_pse_file"] == None:
        #     print(ass_path, info_item["project_dir"], info_item["optimize_level"], info_item["a_ass_pse_file"])
        a_ass_file = os.path.join(ass_path, info_item["project_dir"], info_item["optimize_level"], info_item["a_ass_pse_file"])
        a_pse_file = os.path.join(pse_path, info_item["project_dir"], info_item["optimize_level"], info_item["a_ass_pse_file"])

        # b files
        b_ass_file = os.path.join(ass_path, info_item["project_dir"], info_item["optimize_level"], info_item["b_ass_pse_file"])
        b_pse_file = os.path.join(pse_path, info_item["project_dir"], info_item["optimize_level"], info_item["b_ass_pse_file"])

        # label
        label = info_item["category"]
        label = 1 if label == "security" else 0 

        # 缺少原文件
        if not (os.path.exists(a_ass_file) and os.path.exists(a_pse_file) and os.path.exists(b_ass_file) and os.path.exists(b_pse_file)):
            with open('./errors.log', 'a') as errors:
                id = str(info_item["id"])
                errors.write(f"File(s) of item {id} dose not exist!\n")
            continue
        
        # Step1. 处理pseudo code(因为比较简单...)
        with open(a_pse_file, 'r') as f:
            a_pse = json.load(f)
        a_pse_code = '\n'.join(a_pse["pseudo_code"])
        with open(b_pse_file, 'r') as f:
            b_pse = json.load(f)
        b_pse_code = '\n'.join(b_pse["pseudo_code"])

        # qwen_7b
        # instruction = """
        # [Task]\nHere is a defect function and a fixed function. Please determine if this fixed function fixes any software security vulnerabilities. If it is, output 'yes'. If not, output 'no' and do not output any other text.\n
        # """ + "[Defect function]\n" + a_pse_code + "[Fixed function]\n" + b_pse_code
        
        # qwen3_8b
        instruction = (
            """<|im_start|>system
        You are now an expert in code vulnerability and patch fixes.<|im_end|>
        <|im_start|>user
        [Task]
        Here is a defect function and a fixed function. Please determine if this fixed function fixes any software security vulnerabilities. If it is, output 'yes'. If not, output 'no' and do not output any other text."""
            + "\n[Defect function]\n" + a_pse_code
            + "\n[Fixed function]\n" + b_pse_code
        )


        seq_input = seq_tokenizer(instruction, return_tensors='pt', padding='max_length', max_length=512 ,truncation=True, add_special_tokens=False)

        # Step2. 处理assembly code
        with open(a_ass_file, 'r') as f:
            a_ass = json.load(f)
        with open(b_ass_file, 'r') as f:
            b_ass = json.load(f)

        # diff
        # matches, diffs_a, diffs_b = graph_diff(a_ass, b_ass)
        # G_common = build_graph_from_diff(a_ass, b_ass, matches, diffs_a, diffs_b, mode="common")
        # G_diff   = build_graph_from_diff(a_ass, b_ass, matches, diffs_a, diffs_b, mode="diff")
        # if len(G_common["nodes"]) == 0 or len(G_diff["nodes"]) == 0:
        #     # 兜底回原始 CFG
        #     a_function_nodes_list, a_function_attention_masks_list, a_edges_list, a_edges_type_list = process_graph(a_ass, node_tokenizer)
        #     b_function_nodes_list, b_function_attention_masks_list, b_edges_list, b_edges_type_list = process_graph(b_ass, node_tokenizer)
        # else:
        #     # 记录节点数量比值
        #     ratio = len(G_common["nodes"]) / len(G_diff["nodes"]) if len(G_diff["nodes"]) > 0 else float('inf')

        #     a_function_nodes_list, a_function_attention_masks_list, a_edges_list, a_edges_type_list = process_graph(G_common, node_tokenizer)
        #     b_function_nodes_list, b_function_attention_masks_list, b_edges_list, b_edges_type_list = process_graph(G_diff, node_tokenizer)
        
        a_function_nodes_list, a_function_attention_masks_list, a_edges_list, a_edges_type_list = process_graph(a_ass, node_tokenizer)
        b_function_nodes_list, b_function_attention_masks_list, b_edges_list, b_edges_type_list = process_graph(b_ass, node_tokenizer)
        if len(a_edges_list) == 0:
            a_edges = torch.empty((2, 0), dtype=torch.long)  # 创建一个形状为 [2, 0] 的空张量
        else:
            a_edges = torch.tensor(a_edges_list, dtype=torch.long).permute(1, 0)

        if len(b_edges_list) == 0:
            b_edges = torch.empty((2, 0), dtype=torch.long)  # 创建一个形状为 [2, 0] 的空张量
        else:
            b_edges = torch.tensor(b_edges_list, dtype=torch.long).permute(1, 0)

        inputs = Data(
            a_nodes=torch.tensor(a_function_nodes_list, dtype=torch.long), # list of Tensor
            a_nodes_attention_mask=torch.tensor(a_function_attention_masks_list, dtype=torch.bool), # list of Tensor
            a_edges=a_edges, # list of Tensor
            a_edges_type=torch.tensor(a_edges_type_list, dtype=torch.long), # list of Tensor
            
            b_nodes=torch.tensor(b_function_nodes_list, dtype=torch.long), # list of Tensor
            b_nodes_attention_mask=torch.tensor(b_function_attention_masks_list, dtype=torch.bool), # list of Tensor 
            b_edges=b_edges, 
            b_edges_type=torch.tensor(b_edges_type_list, dtype=torch.long), 
            
            # 添加了a_batch 和 b_batch，用于后续pooler操作
            a_batch = torch.zeros(len(a_function_nodes_list), dtype=torch.long),
            b_batch = torch.zeros(len(b_function_nodes_list), dtype=torch.long),
            instruction = instruction,
            seq_input=seq_input,

            label=torch.tensor(label, dtype=torch.long)
        )
        
        if idx % 100 == 0:
            print(idx)

        inputs_list.append(inputs)

    if (len(inputs_list) == 0):
        print(f'ERROR: Fail to load data from {root_path}')
    else:
        logger.info('finish loading data '+RunTime())
        print("Finishing loading Dataset!")

    return inputs_list

def graph_diff(a_ass, b_ass):
    a_nodes, b_nodes = a_ass["nodes"], b_ass["nodes"]
    a_edges, b_edges = a_ass["edges"], b_ass["edges"]

    matches, diffs_a, diffs_b = [], [], []

    def degree_info(edges, nid):
        indeg = sum([nid in v for v in edges.values()])
        outdeg = len(edges.get(nid, []))
        return indeg, outdeg

    for aid, a_instrs in a_nodes.items():
        best_bid, best_score = None, 0.0
        a_indeg, a_outdeg = degree_info(a_edges, aid)

        for bid, b_instrs in b_nodes.items():
            b_indeg, b_outdeg = degree_info(b_edges, bid)
            if (a_indeg, a_outdeg) != (b_indeg, b_outdeg):
                continue

            sm = difflib.SequenceMatcher(None, " ".join(a_instrs), " ".join(b_instrs))
            score = sm.ratio()
            if score > best_score:
                best_bid, best_score = bid, score

        if best_score > 0.90:  # 阈值
            matches.append((aid, best_bid))
        else:
            diffs_a.append(aid)

    matched_b = {b for _, b in matches}
    for bid in b_nodes.keys():
        if bid not in matched_b:
            diffs_b.append(bid)

    return matches, diffs_a, diffs_b


def build_graph_from_diff(a_ass, b_ass, matches, diffs_a, diffs_b, mode="common"):
    new_graph = {"nodes": {}, "edges": {}, "jmp_instruction": {}, "boundary": {}}

    if mode == "common":
        keep_nodes = {aid for aid, _ in matches} | {bid for _, bid in matches}
    else:
        keep_nodes = set(diffs_a) | set(diffs_b)

    # 处理 nodes
    for nid, instrs in {**a_ass["nodes"], **b_ass["nodes"]}.items():
        if nid in keep_nodes:
            new_graph["nodes"][nid] = instrs

    # 处理 edges
    for src, dsts in {**a_ass["edges"], **b_ass["edges"]}.items():
        if src not in keep_nodes: 
            continue
        new_graph["edges"][src] = [d for d in dsts if d in keep_nodes]

    # 处理 jmp_instruction
    for src, jmp in {**a_ass["jmp_instruction"], **b_ass["jmp_instruction"]}.items():
        if src in keep_nodes:
            new_graph["jmp_instruction"][src] = jmp

    return new_graph


def collect_node_lengths(graphs, node_tokenizer):
    """
    统计所有 node 的编码长度分布
    graphs: list of graph dicts (每个含有 "nodes")
    """
    lengths = []
    for g in graphs:
        for node_id, instrs in g["nodes"].items():
            node = '[CLS]' + str(node_id) + ' ' + ' '.join(instrs)
            tokenized = node_tokenizer(node, return_tensors='pt', add_special_tokens=True, truncation=False)
            length = tokenized["input_ids"].shape[1]
            lengths.append(length)
    return lengths


def analyze_lengths(lengths):
    lengths = np.array(lengths)
    p90 = int(np.percentile(lengths, 90))
    p95 = int(np.percentile(lengths, 95))
    max_len = int(lengths.max())
    avg_len = float(lengths.mean())
    print(f"节点总数: {len(lengths)}")
    print(f"平均长度: {avg_len:.2f}")
    print(f"最大长度: {max_len}")
    print(f"覆盖 90% 节点的长度: {p90}")
    print(f"覆盖 95% 节点的长度: {p95}")
    return p90, p95, max_len
