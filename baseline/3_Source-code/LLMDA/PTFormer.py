import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        assert dim == self.dim, f"Input dimension should be {self.dim}, but got {dim}"

        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.dim // self.num_heads)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.dim // self.num_heads)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.dim // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim // self.num_heads) ** 0.5
        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)
        out = self.out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, patch_embedding, text_embedding):
        batch_size, seq_length, dim = patch_embedding.size()
        q = self.query(patch_embedding)
        k = self.key(text_embedding)
        v = self.value(text_embedding)

        scores = torch.matmul(q, k.transpose(-2, -1)) / dim ** 0.5
        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, v)
        out = self.out(out)
        return out

class PTFormer(nn.Module):
    def __init__(self, patch_dim, text_dim, num_heads, hidden_dim):
        super(PTFormer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(text_dim, num_heads)
        self.cross_attention = CrossAttention(patch_dim)

        self.ff_patch = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, patch_dim)
        )
        
        self.ff_text = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim)
        )

    def forward(self, patch_embedding, explanation_embedding, description_embedding, instruction_embedding):
        # Self-Attention on Text Embeddings
        updated_explanation = self.self_attention(explanation_embedding)
        updated_description = self.self_attention(description_embedding)
        updated_instruction = self.self_attention(instruction_embedding)

        # Cross-Attention between Patch and Explanation
        cross_patch_explanation = self.cross_attention(patch_embedding, updated_explanation)

        # Feedforward for all embeddings
        updated_patch = self.ff_patch(cross_patch_explanation)
        updated_description = self.ff_text(updated_description)
        updated_instruction = self.ff_text(updated_instruction)

        # Concatenate all embeddings
        fused_embedding = torch.cat([updated_patch, updated_description, updated_instruction], dim=-1)
        return fused_embedding

# Example usage
patch_dim = 768
text_dim = 768
num_heads = 8
hidden_dim = 512

pt_former = PTFormer(patch_dim, text_dim, num_heads, hidden_dim)

patch_embedding = torch.randn(32, 100, patch_dim)  # (batch_size, seq_length, dim)
explanation_embedding = torch.randn(32, 100, text_dim)
description_embedding = torch.randn(32, 100, text_dim)
instruction_embedding = torch.randn(32, 100, text_dim)

output = pt_former(patch_embedding, explanation_embedding, description_embedding, instruction_embedding)
print(output.shape)  # Should be (batch_size, seq_length, fused_dim)
