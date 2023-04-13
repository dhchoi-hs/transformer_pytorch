import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_feature, out_feature) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn([in_feature, out_feature]))
        self.bias = nn.Parameter(torch.randn([out_feature,]))
    
    def forward(self, x):
        return x @ self.weights + self.bias

