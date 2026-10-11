import torch
from transformers import AutoTokenizer

def process_graph(a_ass, node_tokenizer):
    
    a_function_nodes_list = [] # list of list
    a_function_attention_masks_list = [] # lol
    a_edges_list = [] # list of tuple
    a_edges_type_list = [] # list
    
    # Step2. 处理assembly code
    block_index = dict() # 用于记录block的index
    
    # Step2-1. 获取每个node的token_ids
    for node_ in a_ass["nodes"]:
        node = '[CLS]' + str(node_) + ' ' + ' '.join(a_ass["nodes"][node_])
        
        # train set: 90% node 的 token 数 < 51, 95% node 的 token 数 < 71
        # test set： 90% node 的 token 数 < 56, 95% node 的 token 数 < 79
        # unixcoder tokenizer

        node = node_tokenizer(node, return_tensors='pt', padding='max_length', max_length=64, truncation=True) 
        a_function_nodes_list.append(list(node["input_ids"][0]))
        a_function_attention_masks_list.append(list(node["attention_mask"][0]))
        
        # 记录当前block的index
        block_index[node_] = len(block_index)

    # Step2-2. 获取edges
    for src in a_ass["edges"]:
        for dst in a_ass["edges"][src]: # 有些node是叶子节点没有出度，则也不考虑edge type，所以将获取et的循环写在这个循环下面
            src, dst = str(src), str(dst) # lqy: 边信息其实就是一些节点地址？那么要不要考虑把这个节点地址也嵌入到节点信息中，使得边信息和节点信息能勾连起来。如："6088": ["cmp 20 3","jg ADDR"]，原来只是读取"cmp 20 3","jg ADDR"，现在把"6088"即地址也加在前面
            
            if (src not in block_index) or (dst not in block_index): continue
            a_edges_list.append([block_index[src], block_index[dst]])
            
            # Step2-3. 获取edges type，无条件跳转1；有条件跳转0
            if 'jmp' in a_ass["jmp_instruction"][src] or 'call' in a_ass["jmp_instruction"][src] or 'retn' in a_ass["jmp_instruction"][src]:
                a_edges_type_list.append(1)
            else:
                a_edges_type_list.append(0)

    assert len(a_edges_list) == len(a_edges_type_list)

    return a_function_nodes_list, a_function_attention_masks_list, a_edges_list, a_edges_type_list
