import torch
import torch.nn as nn
import numpy.polynomial as p
import numpy as np

class Polyformer_Last_Layer(nn.Module):
    def __init__(self, x, time_grid, forecasting_period):
        super(Polyformer_Last_Layer, self).__init__()
        self.degree = input_size

        self.time_vec = time_vec

    def forward(self, coeff):
        coefficients = self.type.chebfit(x)
        return coefficients


class PolyformerEmbedding(nn.Module):
    """
    PolyformerEmbedding module for processing 3D input tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, signal_len, 2).
        signal_len (int, optional): Length of the signal. Default is 128.

    Raises:
        ValueError: If the input tensor does not have the required dimensions.

    Attributes:
        dim (int): Number of dimensions in the input tensor.
        degree (float): Degree parameter for Chebyshev polynomial fitting.
        signal_len (int): Length of the signal.

    """

    def __init__(self, x, signal_len=128):
        super(PolyformerEmbedding, self).__init__()
        self.dim = x.dim()
        self.degree = np.sqrt(signal_len) - 1
        self.signal_len = signal_len

        if self.dim != 3:
            raise ValueError("Input should have dim 3 but has different dimension")

        if x.size(1) != self.signal_len or x.size(2) != 2:
            raise ValueError("Dimension 1 should be {}, and dimension 2 should be 2!".format(signal_len))

    def forward(self, x):
        """
        Forward pass for PolyformerEmbedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, signal_len, 2).

        Returns:
            torch.Tensor: Transformed tensor based on Chebyshev polynomial fitting.

        """
        batch_size, signal_len, _ = x.size()
        batch_tensor = torch.zeros((batch_size, signal_len, int(self.degree) + 1), dtype=x.dtype)

        for batch_idx in range(batch_size):
            counter = -1
            matrix = torch.zeros((signal_len, int(self.degree) + 1), dtype=x.dtype)
            
            while counter < signal_len - 1:
                boolean = True
                counter += 1
                coordinates = torch.zeros(int(self.degree) + 1, dtype=x.dtype)

                while boolean:
                    coordinates[counter] = x[batch_idx, counter, 0]
                    counter = counter + np.sqrt(self.signal_len)

                    if counter + np.sqrt(self.signal_len) > self.signal_len - 1:
                        coefficients = p.chebyshev.chebfit(coordinates, deg=self.degree)
                        boolean = False
                        matrix[counter - int(np.sqrt(self.signal_len)):, :] = coefficients.reshape(1, -1)

            batch_tensor[batch_idx, :, :] = matrix

        return batch_tensor
    

def PolyformerEmbedding_test():
    """
    Example test function for PolyformerEmbedding.

    """
    input_tensor = torch.randn((4, 128, 2))
    print(input_tensor)
    polyformer = PolyformerEmbedding(input_tensor)

    output_tensor = polyformer.forward(input_tensor)
    print(output_tensor)
    print("Example Test Output Shape:", output_tensor.shape)



if __name__=="__main__":
    PolyformerEmbedding_test()



