import torch
import torch.nn as nn

class SimpleActivationLayer(nn.Module):
    """
    A simple fully connected layer with an optional activation function.

    Args:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        activation (callable, optional): Activation function to apply. Defaults to ReLU.
        weight_init (callable, optional): Function to initialize the layer weights. Defaults to None.
    """

    def __init__(self, input_size, output_size, activation=nn.functional.relu, custom_weight_init=None):
        super(SimpleActivationLayer, self).__init__()

        # Learnable parameters
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        self.biases = nn.Parameter(torch.randn(output_size))

        # Activation function
        self.activation = activation

        # Weight initialization
        self.weight_init = custom_weight_init
        if self.weight_init is not None:
            self.weight_init(self.weights)


    def forward(self, x):
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # Perform linear operation: y = wx + b
        affin_linear_output = torch.matmul(x, self.weights.t()) + self.biases

        # Apply activation if specified
        if self.activation is not None:
            affin_linear_output = self.activation(affin_linear_output)

        return affin_linear_output