import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = 1 if item['name'] == 'security' else 0  # 假设非安全类别为0

        # 使用tokenizer处理文本
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 转换为字典，移除添加label到inputs的代码
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, label


def load_data(tokenizer, dataset=None, data_dir=None, train_batch_size=None, test_batch_size=None, model_name=None, method=None, workers=4):
    """
    加载训练和测试数据
    tokenizer: 分词器
    dataset: 数据集名称
    data_dir: 数据目录
    train_batch_size: 训练批次大小
    test_batch_size: 测试批次大小
    model_name: 模型名称
    method: 训练方法
    workers: 数据加载器的工作线程数
    return: 训练数据加载器和测试数据加载器
    """
    # 加载训练数据
    train_path = os.path.join(data_dir, 'train.json')
    with open(train_path, 'r') as f:
        train_data = json.load(f)

    # 加载测试数据
    test_path = os.path.join(data_dir, 'test.json')
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # 创建数据集
    train_dataset = CodeDataset(train_data, tokenizer, max_length=512)
    test_dataset = CodeDataset(test_data, tokenizer, max_length=512)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers
    )

    return train_loader, test_loader