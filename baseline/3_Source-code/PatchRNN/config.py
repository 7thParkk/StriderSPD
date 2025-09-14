"""
Configuration file for Security Patch Identification RNN
Maintains compatibility with original hyperparameters
"""
import torch
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    """Configuration class maintaining original hyperparameters"""
    
    # Environment settings
    colab_mode: bool = False
    debug_mode: bool = False
    
    # File paths
    root_path: str = "./"
    data_path: str = "./data/"
    security_data_path: str = "./data/security_patch/"
    positive_data_path: str = "./data/positives/"
    negative_data_path: str = "./data/negatives/"
    temp_path: str = "./temp/"
    
    # JSON data paths (for mydata)
    mydata_path: str = "./modern_patch_rnn/mydata/"
    train_json_path: str = "./mydata/train.json"
    valid_json_path: str = "./mydata/valid.json"
    test_json_path: str = "./mydata/test.json"
    
    # Hyperparameters - Embedding dimensions (affect GPU memory)
    diff_embed_dim: int = 128
    diff_max_len: int = 600
    diff_rnn_hidden_size: int = 16
    msg_embed_dim: int = 128
    msg_max_len: int = 200
    msg_rnn_hidden_size: int = 16
    twin_embed_dim: int = 128
    twin_max_len: int = 800
    twin_rnn_hidden_size: int = 16
    
    # Hyperparameters - Training speed
    diff_rnn_batch_size: int = 128
    diff_rnn_learn_rate: float = 0.0001
    msg_rnn_batch_size: int = 128
    msg_rnn_learn_rate: float = 0.0001
    patch_rnn_batch_size: int = 256
    patch_rnn_learn_rate: float = 0.0005
    twin_rnn_batch_size: int = 256
    twin_rnn_learn_rate: float = 0.0005
    
    # Network parameters
    diff_extra_dim: int = 6
    twin_extra_dim: int = 1
    diff_rnn_hidden_layers: int = 1
    msg_rnn_hidden_layers: int = 1
    twin_rnn_hidden_layers: int = 1
    
    # Training parameters
    diff_rnn_max_epoch: int = 1000
    diff_rnn_per_epoch: int = 1
    diff_rnn_judge_epoch: int = 10
    msg_rnn_max_epoch: int = 1000
    msg_rnn_per_epoch: int = 1
    msg_rnn_judge_epoch: int = 10
    patch_rnn_max_epoch: int = 1000
    patch_rnn_per_epoch: int = 1
    patch_rnn_judge_epoch: int = 10
    twin_rnn_max_epoch: int = 1000
    twin_rnn_per_epoch: int = 1
    twin_rnn_judge_epoch: int = 10
    
    # Flow control
    lock_split: bool = False  # 0: create random split, 1: use saved split
    load_model: bool = False  # 0: train new model, 1: load saved model
    diff_type: int = 1  # 0: maintain both diff and context, 1: only diff
    comment_type: int = 1  # 0: maintain code and comments, 1: only code
    normalize_identifiers: int = 1  # -1: no abstract, 0: VAR/FUNC, 1: VARn/FUNCn
    normalize_literals: int = 1  # 0: LITERAL, 1: LITERAL/n
    use_twin: int = 1  # 0: only twin network, 1: twins + msg network
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Class settings
    num_classes: int = 2
    vocab_size: int = 10000  
    epochs: int = 5  
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.temp_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

def get_config() -> Config:
    """Get default configuration"""
    return Config()
