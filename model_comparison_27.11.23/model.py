import torch
import torch.nn as nn
from Layer.MultiHeadAttentionLayer import MultiheadAttention
from Layer.SimpleActivationLayer import SimpleActivationLayer
class MyModel(nn.Module):
    def __init__(self, input_size=100, hidden_size1=1000, hidden_size2=100,  hidden_size3=10, output_size=1):
        super(MyModel, self).__init__()

        # Layers
        self.att_layer = MultiheadAttention(input_size, 1)  # 1 head only
        self.layer1 = SimpleActivationLayer(input_size, hidden_size1)
        self.layer2 = SimpleActivationLayer(hidden_size1, hidden_size2)
        self.layer3 = SimpleActivationLayer(hidden_size2, hidden_size3)
        self.linear = nn.Linear(in_features=hidden_size3, out_features=output_size)

    def forward(self, x):
        #print(x.shape)
        #print(x)
        x = self.att_layer(x)
        #print(x)
        x = self.layer1(x)
        #print(x)
        x = self.layer2(x)
        #print(x)
        x = self.layer3(x)
        return self.linear(x)