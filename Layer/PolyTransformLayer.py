import torch.nn as nn
import numpy.polynomial as p

class ChebyTransformlayer(nn.Module):
    def __init__(self, input_size, polynomial_type=p.chebyshev):
        super(ChebyTransformlayer, self).__init__()
        self.type = polynomial_type
        self.degree = input_size
        self.type = polynomial_type

    def forward(self, x):
        coefficients = self.type.chebfit(x)
        return coefficients
    
class inverse_ChebyTransformlayer(nn.Module):
    def __init__(self, input_size, time_vec, polynomial_type=p.chebyshev):
        super(inverse_ChebyTransformlayer, self).__init__()
        self.type = polynomial_type
        self.degree = input_size
        self.type = polynomial_type
        self.time_vec = time_vec

    def forward(self, coeff):
        coefficients = self.type.chebfit(x)
        return coefficients