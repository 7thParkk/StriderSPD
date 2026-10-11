import random
import os
import numpy as np
import torch

def set_seed(seed=44):
    """设置随机种子，确保实验结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 注意拼写：PYTHONHASHSEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
