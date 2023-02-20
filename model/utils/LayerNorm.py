import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, device=None) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(normalized_shape, device=device))
        self.gamma = nn.Parameter(torch.ones(normalized_shape, device=device))
        self.eps = eps
    
    def forward(self, x):
        std = x.std(-1, keepdim=True)
        mean = x.mean(-1, keepdim=True)
        x = ((x - mean) / (std+self.eps)) * self.gamma + self.beta

        return x
