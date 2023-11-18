import torch
import torch.nn.functional as F
import unittest
from Layer.MultiHeadAttentionLayer import MultiheadAttention


class TestMultiheadAttention(unittest.TestCase):
    def setUp(self):
        self.input_size = 64
        self.num_heads = 4
        self.dropout = 0.1
        self.multihead_attention = MultiheadAttention(self.input_size, self.num_heads, self.dropout)

    def test_forward(self):
        # Test forward pass
        input_data = torch.randn(16, 10, self.input_size)
        output_data = self.multihead_attention(input_data)
        self.assertEqual(output_data.shape, (16, 10, self.input_size))

    def test_weight_initialization(self):
        # Test weight initialization
        self.assertTrue(self.check_xavier_uniform(self.multihead_attention.W_q.weight))
        self.assertTrue(self.check_xavier_uniform(self.multihead_attention.W_k.weight))
        self.assertTrue(self.check_xavier_uniform(self.multihead_attention.W_v.weight))
        self.assertTrue(self.check_xavier_uniform(self.multihead_attention.W_o.weight))

    def check_xavier_uniform(self, weight):
        # Check if the weight matrix is initialized with Xavier (Glorot) uniform
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        return torch.allclose(weight, torch.empty_like(weight).uniform_(-bound, bound))

if __name__ == '__main__':
    #unittest.main()

    

    # Example usage
    input_size = 64
    num_heads = 4
    dropout = 0.1

    multihead_attention = MultiheadAttention(input_size, num_heads, dropout)

    # Generate some random input
    input_data = torch.randn(16, 10, input_size)

    # Forward pass
    output_data = multihead_attention(input_data)
    print(output_data.shape)
