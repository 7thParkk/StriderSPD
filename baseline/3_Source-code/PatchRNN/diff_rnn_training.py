#!/usr/bin/env python3
"""
专门训练DiffRNN的脚本 - 使用真实diff数据
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from collections import Counter

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

class DiffRNN(nn.Module):
    """专门用于diff分析的RNN模型"""
    def __init__(self, vocab_size, embed_dim=128, hidden_size=64, extra_dim=6):
        super(DiffRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.extra_fc = nn.Linear(extra_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size * 2 + hidden_size, 2)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2 + hidden_size)
        
    def forward(self, x, extra_features):
        # 文本嵌入
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        last_output = lstm_out[:, -1, :]
        
        # 处理额外特征
        extra_out = torch.relu(self.extra_fc(extra_features))
        
        # 融合特征
        combined = torch.cat([last_output, extra_out], dim=1)
        combined = self.batch_norm(combined)
        output = self.dropout(combined)
        logits = self.classifier(output)
        return logits

def load_json_data():
    """加载JSON数据集"""
    print("📊 加载diff数据集...")
    
    with open('mydata/train.json', 'r') as f:
        train_data = json.load(f)
    with open('mydata/valid.json', 'r') as f:
        valid_data = json.load(f)
    with open('mydata/test.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(valid_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    
    return train_data, valid_data, test_data

def build_vocabulary(train_data, valid_data, test_data, max_vocab_size=15000):
    """构建词汇表"""
    print("🔤 构建diff词汇表...")
    
    all_tokens = []
    
    # 从text字段收集tokens
    for data in [train_data, valid_data, test_data]:
        for item in data:
            if 'text' in item and item['text']:
                text = item['text']
                # 更细致的tokenization
                tokens = text.replace('\n', ' ').replace('\t', ' ').replace('@@', ' @@ ').split()
                all_tokens.extend(tokens)
    
    # 统计词频并构建词汇表
    token_counts = Counter(all_tokens)
    vocab = ['<PAD>', '<UNK>'] + [token for token, count in token_counts.most_common(max_vocab_size-2)]
    
    # 创建token到ID的映射
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    
    print(f"  词汇表大小: {len(vocab)}")
    print(f"  最常见的20个tokens: {list(token_counts.most_common(20))}")
    
    return vocab, token_to_id

def tokenize_data(data, token_to_id, max_len):
    """将文本数据转换为token ID序列"""
    tokenized = []
    for text in data:
        if text is None:
            text = ""
        elif isinstance(text, str):
            tokens = text.replace('\n', ' ').replace('\t', ' ').replace('@@', ' @@ ').split()
        else:
            tokens = []
        
        # 转换为ID
        ids = [token_to_id.get(token, 1) for token in tokens]  # 1是<UNK>的ID
        
        # 截断或填充到固定长度
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids.extend([0] * (max_len - len(ids)))  # 0是<PAD>的ID
            
        tokenized.append(ids)
    
    return np.array(tokenized)

def extract_diff_features(data):
    """提取diff特征"""
    features = []
    
    for item in data:
        text = item.get('text', '')
        
        # 计算diff统计特征
        lines = text.split('\n')
        added_lines = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
        deleted_lines = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
        context_lines = sum(1 for line in lines if not line.startswith(('+', '-', '@', 'diff', 'index')))
        
        feature = [
            len(text),  # 总长度
            added_lines,  # 添加行数
            deleted_lines,  # 删除行数
            context_lines,  # 上下文行数
            text.count('@@'),  # 代码块数量
            len(lines)  # 总行数
        ]
        
        features.append(feature)
    
    return np.array(features, dtype=np.float32)

def prepare_data(train_data, valid_data, test_data, config):
    """准备训练数据"""
    print("🔧 准备diff训练数据...")
    
    # 构建词汇表
    vocab, token_to_id = build_vocabulary(train_data, valid_data, test_data)
    config.vocab_size = len(vocab)
    
    # 提取标签并转换为整数
    def convert_label(label):
        return 1 if (isinstance(label, str) and label.lower() == 'security') else 0
    
    train_labels = np.array([convert_label(item.get('label', 0)) for item in train_data], dtype=int)
    valid_labels = np.array([convert_label(item.get('label', 0)) for item in valid_data], dtype=int)
    test_labels = np.array([convert_label(item.get('label', 0)) for item in test_data], dtype=int)
    
    print(f"  标签分布 - 训练集: {np.bincount(train_labels)} (非安全/安全)")
    print(f"  标签分布 - 验证集: {np.bincount(valid_labels)} (非安全/安全)")
    print(f"  标签分布 - 测试集: {np.bincount(test_labels)} (非安全/安全)")
    
    # 准备diff数据
    train_diff = [item.get('text', '') for item in train_data]
    valid_diff = [item.get('text', '') for item in valid_data]
    test_diff = [item.get('text', '') for item in test_data]
    
    data = {
        'train': (
            torch.LongTensor(tokenize_data(train_diff, token_to_id, config.diff_max_len)),
            torch.FloatTensor(extract_diff_features(train_data))
        ),
        'valid': (
            torch.LongTensor(tokenize_data(valid_diff, token_to_id, config.diff_max_len)),
            torch.FloatTensor(extract_diff_features(valid_data))
        ),
        'test': (
            torch.LongTensor(tokenize_data(test_diff, token_to_id, config.diff_max_len)),
            torch.FloatTensor(extract_diff_features(test_data))
        ),
        'labels': {
            'train': torch.LongTensor(train_labels),
            'valid': torch.LongTensor(valid_labels),
            'test': torch.LongTensor(test_labels)
        }
    }
    
    return data, vocab, token_to_id

def save_test_details(epoch, test_predictions, test_labels, test_probs, model_name="diffrnn"):
    """保存测试详情"""
    os.makedirs("./temp/logs", exist_ok=True)
    
    test_details = []
    for i, (pred, true_label, probs) in enumerate(zip(test_predictions, test_labels, test_probs)):
        pred_label = int(pred)
        true_label = int(true_label)
        
        if true_label == 1 and pred_label == 1:
            pred_type = "TP"
        elif true_label == 0 and pred_label == 0:
            pred_type = "TN"
        elif true_label == 0 and pred_label == 1:
            pred_type = "FP"
        else:
            pred_type = "FN"
        
        detail = {
            "sample_id": i,
            "batch_id": i // 32,
            "batch_position": i % 32,
            "true_label": true_label,
            "predicted_label": pred_label,
            "prediction_probabilities": {
                "class_0_prob": float(probs[0]),
                "class_1_prob": float(probs[1])
            },
            "confidence": float(max(probs)),
            "is_correct": pred_label == true_label,
            "prediction_type": pred_type
        }
        test_details.append(detail)
    
    filename = f"./temp/logs/{model_name}_epoch_{epoch:03d}_test_details.json"
    with open(filename, 'w') as f:
        json.dump(test_details, f, indent=2)
    
    print(f"  ✓ 测试详情已保存: {filename}")
    return filename

def train_diff_rnn(model, data, epochs=100, lr=0.001):
    """训练DiffRNN模型"""
    print(f"\n🔥 开始训练 DiffRNN...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    batch_size = 64
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 早停机制
    best_valid_acc = 0
    best_test_acc = 0
    patience = 15
    patience_counter = 0
    
    train_data, train_extra = data['train']
    valid_data, valid_extra = data['valid']
    test_data, test_extra = data['test']
    train_labels = data['labels']['train']
    valid_labels = data['labels']['valid']
    test_labels = data['labels']['test']
    
    for epoch in range(epochs):
        model.train()
        
        # 训练
        train_loss = 0
        train_correct = 0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_extra = train_extra[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_data, batch_extra)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == batch_labels).sum().item()
            num_batches += 1
        
        train_acc = train_correct / len(train_labels)
        avg_train_loss = train_loss / num_batches
        
        # 验证
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_data, valid_extra)
            _, valid_predicted = torch.max(valid_outputs.data, 1)
            valid_acc = (valid_predicted == valid_labels).sum().item() / len(valid_labels)
            
            # 测试并保存详情
            test_outputs = model(test_data, test_extra)
            test_probs = torch.softmax(test_outputs, dim=1)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == test_labels).sum().item() / len(test_labels)
            
            # 保存测试详情
            save_test_details(
                epoch + 1,
                test_predicted.numpy(),
                test_labels.numpy(),
                test_probs.numpy(),
                "diffrnn"
            )
        
        print(f"  Epoch {epoch+1:3d}/{epochs}: Loss: {avg_train_loss:.4f} | Train: {train_acc:.2%} | Valid: {valid_acc:.2%} | Test: {test_acc:.2%}")
        
        # 学习率调度
        scheduler.step(valid_acc)
        
        # 早停检查
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), './temp/diffrnn_best.pth')
            print(f"  🎯 新的最佳模型! 验证: {best_valid_acc:.2%}, 测试: {best_test_acc:.2%}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  ⏹️ 早停触发! 最佳验证准确率: {best_valid_acc:.2%}, 对应测试准确率: {best_test_acc:.2%}")
            break
    
    print(f"✅ DiffRNN 训练完成!")
    print(f"📊 最终结果: 最佳验证准确率: {best_valid_acc:.2%}, 对应测试准确率: {best_test_acc:.2%}")
    return model

def main():
    print("🚀 DiffRNN 专项训练")
    print("="*60)
    
    # 配置
    config = Config()
    config.diff_max_len = 1000  # 增加序列长度以处理长diff
    epochs = 100
    
    print(f"设备: {config.device}")
    print(f"最大训练轮数: {epochs}")
    print(f"Diff最大长度: {config.diff_max_len}")
    
    # 加载数据
    train_data, valid_data, test_data = load_json_data()
    
    # 准备数据
    data, vocab, token_to_id = prepare_data(train_data, valid_data, test_data, config)
    
    print(f"最终词汇表大小: {config.vocab_size}")
    
    # 创建模型
    model = DiffRNN(
        vocab_size=config.vocab_size,
        embed_dim=config.diff_embed_dim,
        hidden_size=config.diff_rnn_hidden_size * 2,  # 增加隐藏层大小
        extra_dim=6
    )
    
    # 训练模型
    trained_model = train_diff_rnn(model, data, epochs)
    
    # 检查生成的文件
    print("\n📁 检查生成的文件...")
    log_dir = "./temp/logs"
    if os.path.exists(log_dir):
        all_files = os.listdir(log_dir)
        diff_files = [f for f in all_files if 'diffrnn' in f and 'test_details' in f]
        print(f"DiffRNN测试详情文件: {len(diff_files)}")
        
        if diff_files:
            print("✅ DiffRNN测试详情文件已生成!")
            for f in sorted(diff_files)[-5:]:  # 显示最新的5个文件
                filepath = os.path.join(log_dir, f)
                size = os.path.getsize(filepath)
                print(f"  {f}: {size:,} 字节")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()