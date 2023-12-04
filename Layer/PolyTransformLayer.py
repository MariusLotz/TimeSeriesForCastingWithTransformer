import torch
import torch.nn as nn
import numpy.polynomial as p
import numpy as np

class PolyformerInverseEmbedding(nn.Module):
    def __init__(self, x, time_grid, forecasting_length):
        super(PolyformerInverseEmbedding, self).__init__()
        self.time_grid = time_grid
        self.forcasting_grid = time_grid + forecasting_length

    def forward(self, x):
        batch_size, matrix_size, matrix_size = x.size()
        batch_tensor_list = []
        for batch_idx in range(batch_size):
            matrix = x[batch_idx]
            matrix = []
            for i in range(matrix_size):
                time_points = [self.forcasting_grid[i + j * matrix_size] for j in range(matrix_size)]
                pred_values = p.chebyshev.chebval(time_points, matrix[i,:])
                matrix.append([[x, y] for x, y in zip(time_points, pred_values)])
            matrix_tensor = torch.tensor(matrix)
            right_order_matrix = torch.inverse(matrix_tensor)
        batch_tensor_list.append(right_order_matrix)
    
        return batch_tensor_list

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
        self.signal_len = signal_len
        self.degree = int(np.sqrt(signal_len) - 1)
        print(self.degree)
       

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
        batch_size = x.size(0)
        matrix_size = int(np.sqrt(self.signal_len))
        batch_tensor_list = []
        for batch_idx in range(batch_size):
            signal = x[batch_idx]
            for i in range(matrix_size):
                part_signal = []
                tensor_list = []
                for j in range(matrix_size):
                    index = i + j * matrix_size
                    part_signal.append(signal[index])
            tensor_list.append(torch.tensor(p.chebyshev.chebfit(part_signal[:][0], part_signal[:][1], self.degree)))
            batch_tensor_list.append(torch.stack(tensor_list, dim=0))
        batch_matrix_tensor = torch.stack(batch_tensor_list, dim=0)

        return batch_matrix_tensor
    

def PolyformerEmbedding_test():
    """
    Example test function for PolyformerEmbedding.

    """

    time_grids = torch.tensor([[1,2,3,4], [5,6,7,8]])
    values = torch.tensor([[1,2,3,4,], [1,2,3,4]])

    input_tensor = torch.cat([torch.unsqueeze(time_grids, 2), torch.unsqueeze(values, 2)], dim=2)

    print("Input tensor size:", input_tensor.size())
    print("Input tensor:")
    print(input_tensor)

    polyformer = PolyformerEmbedding(input_tensor, signal_len=4)

    output_tensor = polyformer.forward(input_tensor)
    print("Output tensor size:", output_tensor.size())
    print("Output tensor:")
    print(output_tensor)

    reverse_polyformer = PolyformerInverseEmbedding(output_tensor, time_grids, 0)
    reverse_tensor = reverse_polyformer.forward(output_tensor)

    print("reverse_tensor:", reverse_tensor.size())
    print("reverse_tensor:")
    print(reverse_tensor)



if __name__=="__main__":
    PolyformerEmbedding_test()


