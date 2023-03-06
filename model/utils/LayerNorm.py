import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6, device=None) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features, device=device))
        self.beta = nn.Parameter(torch.zeros(features, device=device))
        self.eps = eps
    
    def forward(self, x):
        std = x.std(-1, keepdim=True)
        mean = x.mean(-1, keepdim=True)
        x = ((x - mean) / (std+self.eps)) * self.gamma + self.beta

        return x
