import torch
import torch.nn as nn
from config.logger import setup_logger
logger = setup_logger()

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        )

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q_input, kv_input):

        B, q_len, D = q_input.size()
        kv_len = kv_input.size(1)

        Q = self.q_proj(q_input).view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, q_len, d_h]
        K = self.k_proj(kv_input).view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, kv_len, d_h]
        V = self.v_proj(kv_input).view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, q_len, kv_len]
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # attention * V
        out = attn @ V 
        out = out.transpose(1, 2).reshape(B, q_len, D)

        out = self.proj_drop(self.out_proj(out))
        out = self.norm(out + q_input) 

        return out