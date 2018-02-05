import torch
from torch import nn
from torch.nn.parameter import Parameter

class ScaleLayer(nn.Module):
    def __init__(self, parameters_dimensions=(1, 1, 1, 1), init_value=1):
        super().__init__()
        self.scale = Parameter(torch.ones(parameters_dimensions)*init_value)

    def forward(self, input):
        return input*self.scale
