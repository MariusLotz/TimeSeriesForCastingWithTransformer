import torch.nn as nn
#from ...Layer.MultiHeadAttentionLayer import MultiheadAttention
#from ...Layer.SimpleActivationLayer import SimpleActivationLayer
#from ...Layer.PolyTransformLayer import PolyformerEmbedding
from . import Layer.MultiHeadAttentionLayer.MultiheadAttention

class MyModel(nn.Module):
    def __init__(self,  x, time_grid=[1.0, 2.0, ..., 256.0], forecasting_length=256, degree=15):
        super(MyModel, self).__init__()

        self.time_grid = time_grid
        self.forcasting_length = forecasting_length
        self.degree = degree

        # Layers
        self.embedding_layer = PolyformerEmbedding(self.time_grid, self.forcasting_length, self.degree)
        self.inverse_embedding_layer = PolyformerEmbedding(self.time_grid, self.forcasting_length, self.degree), True
        self.attention_layer = MultiheadAttention(16, 1)  # 1 head only
        self.layer1 = SimpleActivationLayer(16, 16)
        self.layer2 = SimpleActivationLayer(16, 16)
        self.layer3 = SimpleActivationLayer(16, 16)
        self.linear = nn.Linear(in_features=16, out_features=16)
    

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.linear
        x = self.attention_layer(x)
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        x = self.inverse_embedding_layer(x)
        return self.linear(x)