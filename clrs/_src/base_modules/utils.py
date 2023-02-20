import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor, ceil

class LayerNorm(nn.Module):
    def __init__(self, size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, params=None, bias=0.):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        nn.init.constant_(self.layer_bias, bias)

    def forward(self, input):
        output = self.layer(input)
        return output

class ResidualLayer(nn.Module):
    def __init__(self, input_size, output_size, params=None):
        super(ResidualLayer, self).__init__()
        self.linear_layer = LinearLayer(input_size, output_size)

    def forward(self, input, prev_input):
        output = self.linear_layer(input)
        output += prev_input
        return output
