import torch
import torch.nn.functional as F
import unittest
from Layer.MultiHeadAttentionLayer import MultiheadAttention

# Example usage
input_size = 12
num_heads = 4
dropout = 0.1

multihead_attention = MultiheadAttention(input_size, num_heads, dropout)

# Generate some random input
input_data = torch.randn(16, 10, input_size)

# Forward pass
output_data = multihead_attention(input_data)
print(output_data.shape)
