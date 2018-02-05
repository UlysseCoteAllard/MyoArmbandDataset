import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from torch.nn.modules.activation import  ELU

class pelu(nn.Module):
    def __init__(self, parameters_dimensions=(1, 1, 1, 1), alpha=1., beta=1.):
        super(pelu, self).__init__()
        self._alpha = Parameter(torch.ones(parameters_dimensions)*alpha)
        self._beta = Parameter(torch.ones(parameters_dimensions)*beta)

    def forward(self, x):
        # Check if alpha or beta are lower than .1, if so set them to .1
        self._alpha.data -= (self._alpha.data < .1).float()*(self._alpha.data - .1)

        self._beta.data -= (self._beta.data < .1).float()*(self._beta.data - .1)
        return self.where((x >= 0).float(), (self._alpha/self._beta)*x, self._alpha*(torch.exp(x/self._beta)-1))

    def where(self, cond, x_1, x_2):
        return (cond * x_1) + ((1 - cond) * x_2)
