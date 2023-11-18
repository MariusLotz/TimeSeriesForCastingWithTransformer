class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MyModel, self).__init__()

        # Layers
        self.att_layer = MultiheadAttention(input_size, 1)
        self.layer2 = CustomLinear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.layer3 = CustomLinear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x