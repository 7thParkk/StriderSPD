#!/usr/bin/env python3
"""
推理脚本：使用训练好的最佳模型对测试集进行推理
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from models.diff_rnn import DiffRNN
from data.data_loader import load_test_data, create_data_loader

class ModelInference:
    def __init__(self, model_path, config_path=None):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径（可选）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        
    def load_config(self):
        """加载模型配置"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 默认配置（基于训练日志中的参数）
            self.config = {
                'vocab_size': 10000,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'num_classes': 2
            }
        print(f"使用配置: {self.config}")
        
    def load_model(self):
        """加载训练好的模型"""
        self.load_config()
        
        # 创建模型实例
        self.model = DiffRNN(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            num_classes=self.config['num_classes']
        )
        
        # 加载模型权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"加载模型权重，epoch: {checkpoint.get('epoch', 'unknown')}")
                    print(f"验证准确率: {checkpoint.get('val_accuracy', 'unknown')}")
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            print(f"成功加载模型: {self.model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
    
    def predict_batch(self, data_loader):
        """
        批量预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            predictions: 预测结果列表
            probabilities: 预测概率列表
            true_labels: 真实标签列表（如果有的话）
        """
        predictions = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if len(batch) == 2:
                    inputs, labels = batch
                    true_labels.extend(labels.cpu().numpy())
                else:
                    inputs = batch[0]
                    labels = None
                
                inputs = inputs.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 获取预测概率
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
                
                # 获取预测类别
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                
                if batch_idx % 100 == 0:
                    print(f"已处理批次: {batch_idx + 1}")
        
        return predictions, probabilities, true_labels
    
    def save_results(self, predictions, probabilities, true_labels=None, 
                    output_path="inference_results.json"):
        """
        保存推理结果
        
        Args:
            predictions: 预测结果
            probabilities: 预测概率
            true_labels: 真实标签（可选）
            output_path: 输出文件路径
        """
        results = {
            'metadata': {
                'model_path': self.model_path,
                'inference_time': datetime.now().isoformat(),
                'total_samples': len(predictions),
                'device': str(self.device),
                'config': self.config
            },
            'results': []
        }
        
        for i in range(len(predictions)):
            result_item = {
                'sample_id': i,
                'predicted_class': int(predictions[i]),
                'predicted_label': 'vulnerable' if predictions[i] == 1 else 'safe',
                'probability_safe': float(probabilities[i][0]),
                'probability_vulnerable': float(probabilities[i][1]),
                'confidence': float(max(probabilities[i]))
            }
            
            if true_labels:
                result_item['true_class'] = int(true_labels[i])
                result_item['true_label'] = 'vulnerable' if true_labels[i] == 1 else 'safe'
                result_item['correct'] = bool(predictions[i] == true_labels[i])
            
            results['results'].append(result_item)
        
        # 计算统计信息
        if true_labels:
            correct_predictions = sum(1 for i in range(len(predictions)) 
                                    if predictions[i] == true_labels[i])
            accuracy = correct_predictions / len(predictions)
            results['metadata']['accuracy'] = accuracy
            results['metadata']['correct_predictions'] = correct_predictions
            
            # 计算混淆矩阵
            tp = sum(1 for i in range(len(predictions)) 
                    if predictions[i] == 1 and true_labels[i] == 1)
            tn = sum(1 for i in range(len(predictions)) 
                    if predictions[i] == 0 and true_labels[i] == 0)
            fp = sum(1 for i in range(len(predictions)) 
                    if predictions[i] == 1 and true_labels[i] == 0)
            fn = sum(1 for i in range(len(predictions)) 
                    if predictions[i] == 0 and true_labels[i] == 1)
            
            results['metadata']['confusion_matrix'] = {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            }
            
            # 计算精确率、召回率、F1分数
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['metadata']['metrics'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"推理结果已保存到: {output_path}")
        
        # 打印摘要
        print("\n=== 推理结果摘要 ===")
        print(f"总样本数: {len(predictions)}")
        print(f"预测为安全的样本: {sum(1 for p in predictions if p == 0)}")
        print(f"预测为漏洞的样本: {sum(1 for p in predictions if p == 1)}")
        
        if true_labels:
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")

def main():
    """主函数"""
    # 根据训练日志，最佳模型应该在epoch 8
    model_path = "training/checkpoints/best_model.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        # 尝试其他可能的路径
        possible_paths = [
            "training/best_model.pth",
            "models/best_model.pth",
            "checkpoints/best_model.pth",
            "training/checkpoints/diff_rnn_epoch_8.pth",
            "training/diff_rnn_best.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            print("错误: 找不到模型文件")
            print("请确保以下路径之一存在模型文件:")
            for path in possible_paths:
                print(f"  - {path}")
            return
    
    print(f"使用模型文件: {model_path}")
    
    # 创建推理器
    inference = ModelInference(model_path)
    
    try:
        # 加载模型
        inference.load_model()
        
        # 加载测试数据
        print("加载测试数据...")
        test_loader = load_test_data(batch_size=32)
        
        # 进行推理
        print("开始推理...")
        predictions, probabilities, true_labels = inference.predict_batch(test_loader)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"inference_results_{timestamp}.json"
        inference.save_results(predictions, probabilities, true_labels, output_path)
        
        print("推理完成!")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()