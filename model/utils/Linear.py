import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_feature, out_feature) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros([in_feature, out_feature]))
        self.bias = nn.Parameter(torch.zeros([out_feature,]))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))
        nn.init.uniform_(self.bias, -1, 1)
    
    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias

