import torch
import torch.nn as nn
import numpy as np


class Attention_Square_Reduction_Layer(nn.Module):
    def __init__(self, inverse=False):
        super(Attention_Square_Reduction_Layer, self).__init__()
        self.inverse = inverse
        
    def forward(self, x):
        # Reshape the tensor
        if not self.inverse:
            reshaped_x = x.view(x.size(0), int(x.size(1)**0.5), int(x.size(1)**0.5)).transpose(1, 2)
        else:
            reshaped_x = x.transpose(1, 2).view(x.size(0), x.size(1)**2)
        return reshaped_x


def Attention_Square_Reduction_Layer_test():
    """
    Example test function for Attention_Square_Reduction_Layer.

    """
    values = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])

    layer1 = Attention_Square_Reduction_Layer()
    layer2 = Attention_Square_Reduction_Layer(True)

    out = layer2(layer1(values))
    print(layer1(values))
    print(values)
    print(out)


 
if __name__ == "__main__":
    Attention_Square_Reduction_Layer_test()