import torch.nn as nn
from Layer.MultiHeadAttentionLayer import MultiheadAttention
from Layer.SimpleActivationLayer import SimpleActivationLayer
from Layer.PolyTransformLayer import PolyformerEmbedding


class MyModel(nn.Module):
    def __init__(self, time_grid=[(i) for i in range(256)], forecasting_length=256.0, degree=15):
        super(MyModel, self).__init__()

        self.time_grid = time_grid
        self.forcasting_length = forecasting_length
        self.degree = degree

        # Layers
        self.embedding_layer = PolyformerEmbedding(self.time_grid, self.forcasting_length, self.degree)
        self.inverse_embedding_layer = PolyformerEmbedding(self.time_grid, self.forcasting_length, self.degree, True)
        self.attention_layer = MultiheadAttention(256, 1)  # 1 head only
        self.layer1 = SimpleActivationLayer(256, 256)
        self.layer2 = SimpleActivationLayer(256, 256)
        self.layer3 = SimpleActivationLayer(16, 16)
        self.linear1 = nn.Linear(in_features=16, out_features=16)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
    

    def forward(self, x):
        
        #x = self.embedding_layer(x.float())
        
        #x = self.linear1(x.float())
        x = self.attention_layer(x.float())
        x = self.layer1(x.float())
        #x = self.layer2(x.float())
        #x = self.layer3(x.float())
        #x = self.inverse_embedding_layer(x.float())
        return self.linear2(x.float())