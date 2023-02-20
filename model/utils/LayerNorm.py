import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        std = x.std(-1, keepdim=True)
        mean = x.mean(-1, keepdim=True)
        x = ((x - mean) / (std+self.eps)) * self.gamma + self.beta

        return x


i = torch.tensor([[1,2,3,4,5],[6,7,8,9,10.]])

aa = LayerNorm(i.size())
print(i.std(-1), i.mean(-1))

o = aa(i)
print(o.std(-1), o.mean(-1))
