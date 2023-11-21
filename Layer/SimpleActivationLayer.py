import torch
import torch.nn as nn

class SimpleActivationLayer(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.functional.relu, weight_init=None):
        super(SimpleActivationLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        self.biases = nn.Parameter(torch.randn(output_size))
        self.activation = activation
        self.weight_init = weight_init

        if self.weight_init is not None:
            self.weight_init(self.weights)

    def forward(self, x):
        # Perform linear operation: y = wx + b
        linear_output = torch.matmul(x, self.weights.t()) + self.biases

        # Apply activation if specified
        if self.activation is not None:
            linear_output = self.activation(linear_output)

        return linear_output