#!/usr/bin/env python3
"""
完整的PatchRNN训练脚本 - 训练所有模型
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.msg_rnn import MsgRNN, MsgRNNTrainer
from models.diff_rnn import DiffRNN, DiffRNNTrainer
from models.patch_rnn import PatchRNN, PatchRNNTrainer
from models.twin_rnn import TwinRNN, TwinRNNTrainer

def create_synthetic_data(config):
    """创建合成训练数据"""
    print("📊 创建合成训练数据...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    
    # 为不同模型创建不同格式的数据
    data = {}
    
    # MsgRNN数据 (消息数据)
    msg_data = np.random.randint(0, config.vocab_size, (n_samples, config.msg_max_len))
    data['msg'] = msg_data
    
    # DiffRNN数据 (差异数据 + 额外特征)
    diff_data = np.random.randint(0, config.vocab_size, (n_samples, config.diff_max_len))
    diff_extra = np.random.randn(n_samples, config.diff_extra_dim)
    data['diff'] = diff_data
    data['diff_extra'] = diff_extra
    
    # TwinRNN数据 (双胞胎网络数据 + 额外特征)
    twin_data = np.random.randint(0, config.vocab_size, (n_samples, config.twin_max_len))
    twin_extra = np.random.randn(n_samples, config.twin_extra_dim)
    data['twin'] = twin_data
    data['twin_extra'] = twin_extra
    
    # 标签
    labels = np.random.randint(0, 2, n_samples)
    data['labels'] = labels
    
    # 数据集划分
    train_size = int(0.7 * n_samples)
    valid_size = int(0.15 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:train_size]
    valid_idx = indices[train_size:train_size + valid_size]
    test_idx = indices[train_size + valid_size:]
    
    splits = {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx
    }
    
    print(f"  训练集: {len(train_idx)} 样本")
    print(f"  验证集: {len(valid_idx)} 样本")
    print(f"  测试集: {len(test_idx)} 样本")
    
    return data, splits

def train_msg_rnn(config, data, splits):
    """训练MsgRNN模型"""
    print("\n🔥 训练 MsgRNN...")
    
    msg_data = data['msg']
    labels = data['labels']
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']
    
    # 创建训练器
    trainer = MsgRNNTrainer(config)
    trainer.vocab_size = config.vocab_size
    
    try:
        # 训练模型
        model = trainer.train(
            msg_data[train_idx], labels[train_idx],
            msg_data[valid_idx], labels[valid_idx],
            msg_data[test_idx], labels[test_idx]
        )
        print("✅ MsgRNN 训练完成!")
        return model
    except Exception as e:
        print(f"❌ MsgRNN 训练失败: {e}")
        return None

def train_diff_rnn(config, data, splits):
    """训练DiffRNN模型"""
    print("\n🔥 训练 DiffRNN...")
    
    diff_data = data['diff']
    diff_extra = data['diff_extra']
    labels = data['labels']
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']
    
    # 创建训练器
    trainer = DiffRNNTrainer(config)
    trainer.vocab_size = config.vocab_size
    
    try:
        # 训练模型
        model = trainer.train(
            diff_data[train_idx], diff_extra[train_idx], labels[train_idx],
            diff_data[valid_idx], diff_extra[valid_idx], labels[valid_idx],
            diff_data[test_idx], diff_extra[test_idx], labels[test_idx]
        )
        print("✅ DiffRNN 训练完成!")
        return model
    except Exception as e:
        print(f"❌ DiffRNN 训练失败: {e}")
        return None

def train_twin_rnn(config, data, splits):
    """训练TwinRNN模型"""
    print("\n🔥 训练 TwinRNN...")
    
    twin_data = data['twin']
    twin_extra = data['twin_extra']
    labels = data['labels']
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']
    
    # 创建训练器
    trainer = TwinRNNTrainer(config)
    trainer.vocab_size = config.vocab_size
    
    try:
        # 训练模型
        model = trainer.train(
            twin_data[train_idx], twin_extra[train_idx], labels[train_idx],
            twin_data[valid_idx], twin_extra[valid_idx], labels[valid_idx],
            twin_data[test_idx], twin_extra[test_idx], labels[test_idx]
        )
        print("✅ TwinRNN 训练完成!")
        return model
    except Exception as e:
        print(f"❌ TwinRNN 训练失败: {e}")
        return None

def train_patch_rnn(config, msg_model, diff_model, twin_model, data, splits):
    """训练PatchRNN模型（集成模型）"""
    print("\n🔥 训练 PatchRNN (集成模型)...")
    
    if not all([msg_model, diff_model, twin_model]):
        print("❌ 需要先训练好所有子模型才能训练PatchRNN")
        return None
    
    # 创建训练器
    trainer = PatchRNNTrainer(config)
    
    try:
        # 准备数据
        msg_data = data['msg']
        diff_data = data['diff']
        diff_extra = data['diff_extra']
        twin_data = data['twin']
        twin_extra = data['twin_extra']
        labels = data['labels']
        train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']
        
        # 训练集成模型
        model = trainer.train(
            msg_data[train_idx], diff_data[train_idx], diff_extra[train_idx],
            twin_data[train_idx], twin_extra[train_idx], labels[train_idx],
            msg_data[valid_idx], diff_data[valid_idx], diff_extra[valid_idx],
            twin_data[valid_idx], twin_extra[valid_idx], labels[valid_idx],
            msg_data[test_idx], diff_data[test_idx], diff_extra[test_idx],
            twin_data[test_idx], twin_extra[test_idx], labels[test_idx],
            msg_model, diff_model, twin_model
        )
        print("✅ PatchRNN 训练完成!")
        return model
    except Exception as e:
        print(f"❌ PatchRNN 训练失败: {e}")
        return None

def check_generated_files():
    """检查生成的文件"""
    print("\n📁 检查生成的文件...")
    
    log_dir = "./temp/logs"
    if not os.path.exists(log_dir):
        print("❌ 日志目录不存在")
        return
    
    all_files = os.listdir(log_dir)
    test_detail_files = [f for f in all_files if 'test_details' in f]
    
    print(f"总文件数: {len(all_files)}")
    print(f"测试详情文件: {len(test_detail_files)}")
    
    if test_detail_files:
        print("✅ 测试详情文件已生成:")
        for f in sorted(test_detail_files):
            filepath = os.path.join(log_dir, f)
            size = os.path.getsize(filepath)
            print(f"  {f}: {size} 字节")
    else:
        print("❌ 未找到测试详情文件")

def main():
    print("🚀 PatchRNN 完整训练流程")
    print("="*60)
    
    # 配置
    config = Config()
    config.epochs = 5  # 训练5轮
    config.msg_rnn_max_epoch = 5
    config.diff_rnn_max_epoch = 5
    config.twin_rnn_max_epoch = 5
    config.patch_rnn_max_epoch = 5
    config.vocab_size = 1000  # 使用较小的词汇表进行快速训练
    
    print(f"设备: {config.device}")
    print(f"训练轮数: {config.epochs}")
    print(f"词汇表大小: {config.vocab_size}")
    
    # 创建数据
    data, splits = create_synthetic_data(config)
    
    # 存储训练好的模型
    models = {}
    
    # 1. 训练MsgRNN
    models['msg'] = train_msg_rnn(config, data, splits)
    
    # 2. 训练DiffRNN
    models['diff'] = train_diff_rnn(config, data, splits)
    
    # 3. 训练TwinRNN
    models['twin'] = train_twin_rnn(config, data, splits)
    
    # 4. 训练PatchRNN (集成模型)
    models['patch'] = train_patch_rnn(
        config, models['msg'], models['diff'], models['twin'], data, splits
    )
    
    # 检查结果
    print("\n📊 训练结果总结:")
    print("="*40)
    for model_name, model in models.items():
        status = "✅ 成功" if model is not None else "❌ 失败"
        print(f"{model_name.upper()}RNN: {status}")
    
    # 检查生成的文件
    check_generated_files()
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()