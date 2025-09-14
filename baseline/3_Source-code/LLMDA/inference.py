import torch
import json
import os
from model import Transformer
from data_utils import CodeDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    # 配置参数
    model_path = 'bestmodel.h5'  # 模型保存路径
    data_dir = 'data'  # 数据目录
    model_name = 'bert'  # 或 'roberta'，根据训练时使用的模型
    num_classes = 2  # 类别数量
    method = 'ce'  # 训练方法，需要与训练时一致
    max_length = 512  # 文本最大长度
    batch_size = 32  # 推理批次大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer
    if model_name == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        base_model = AutoModel.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)
    elif model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
        base_model = AutoModel.from_pretrained('roberta-base', ignore_mismatched_sizes=True)
    else:
        raise ValueError('unknown model')
    
    # 创建模型并加载权重
    model = Transformer(base_model, num_classes, method)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载测试数据
    test_path = os.path.join(data_dir, 'test.json')
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    # 创建测试数据集和数据加载器
    test_dataset = CodeDataset(test_data, tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 推理并保存结果
    all_predicts = []
    all_labels = []
    all_feats = []
    all_texts = [item['text'] for item in test_data]
    all_names = [item['name'] for item in test_data]
    
    print("开始推理...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)
            
            outputs = model(inputs)
            predicts = torch.argmax(outputs['predicts'], -1)
            
            all_predicts.extend(predicts.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_feats.extend(outputs['cls_feats'].cpu().numpy().tolist())
    
    # 保存推理结果
    results = [{
        'text': all_texts[i],
        'name': all_names[i],
        'true_label': int(all_labels[i]),
        'predicted_label': int(all_predicts[i]),
        'features': all_feats[i]
    } for i in range(len(all_texts))]
    
    with open('inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predicts)
    precision = precision_score(all_labels, all_predicts)
    recall = recall_score(all_labels, all_predicts)
    f1 = f1_score(all_labels, all_predicts)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predicts).ravel()
    
    # 输出summary
    print("\n=== 推理结果 Summary ===")
    print(f"模型: {model_name}")
    print(f"测试样本数: {len(all_labels)}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"混淆矩阵:")
    print(f"    预测安全    预测非安全")
    print(f"实际安全    {tp}          {fn}")
    print(f"实际非安全  {fp}          {tn}")
    print(f"结果已保存至: inference_results.json")

if __name__ == '__main__':
    main()
