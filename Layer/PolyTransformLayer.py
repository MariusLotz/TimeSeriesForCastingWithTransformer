import torch
import torch.nn as nn
import numpy as np
import numpy.polynomial as p

class PolyformerEmbedding(nn.Module):
    """
    PolyformerEmbedding is a PyTorch Module that transforms time-series data using Chebyshev polynomials.

    Attributes:
        time_grid (torch.Tensor): Time grid for the input data.
        forecasting_length (float): Length of forecasting to be applied.
        forecasting_grid (list): Grid for forecasting based on time_grid and forecasting_length.
        inverse (bool): If True, performs the inverse operation.
        degree (int): Degree of the Chebyshev polynomials to be used.

    """

    def __init__(self, time_grid, forecasting_length, degree, inverse=False):
        super(PolyformerEmbedding, self).__init__()
        self.time_grid = time_grid
        self.forecasting_length = forecasting_length
        self.forecasting_grid = [t + self.forecasting_length for t in self.time_grid]
        self.inverse = inverse
        self.degree = degree

    def forward(self, x):
        """
        Forward pass of the PolyformerEmbedding.

        Args:
            x (torch.Tensor): Input tensor representing time-series data.

        Returns:
            torch.Tensor: Transformed tensor based on Chebyshev polynomials.

        AND vice versa
        """
        if not self.inverse:
            matrix_size = int(np.sqrt(x.size(1)))
            batch = []
            for batch_index in range(x.size(0)):
                signal = x[batch_index]
                matrix = []
                for i in range(matrix_size):
                    values = [signal[i + j * matrix_size] for j in range(matrix_size)]
                    times = [self.time_grid[i + j * matrix_size] for j in range(matrix_size)]
                    coeffs = p.chebyshev.chebfit(times, values, self.degree)
                    matrix.append(coeffs)
                batch.append(matrix)
        else:
            matrix_size = x.size(1)
            batch = []
            for batch_index in range(x.size(0)):
                matrix = x[batch_index]
                signal = []
                for j in range(matrix_size):
                    for i in range(matrix_size):
                        time = self.forecasting_grid[i + j * matrix_size]
                        value = p.chebyshev.chebval(time, matrix[i].detach().numpy())
                        signal.append(value)
                batch.append(signal)
       
        return torch.tensor(batch)


def PolyformerEmbedding_test():
    """
    Example test function for PolyformerEmbedding.

    """
    time_grid = torch.tensor([1, 2, 3, 4])
    values = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    print(values)

    polyformer = PolyformerEmbedding(values, time_grid, 0, 1)

    output = polyformer.forward(values)
    # print(output)

    reverse_polyformer = PolyformerEmbedding(output, time_grid, 0.5, 1, True)
    reverse_tensor = reverse_polyformer.forward(output)

    print(reverse_tensor)

if __name__ == "__main__":
    PolyformerEmbedding_test()