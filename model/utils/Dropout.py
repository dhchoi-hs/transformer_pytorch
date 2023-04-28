import torch
from torch import nn


class Dropout(nn.Module):
    def __init__(self, p=0.1) -> None:
        super().__init__()
        assert p < 0 or p > 1, '[ERROR] Invalid value of dropout!'
        self.prob = p

    def forward(self, x):
        if self.training and self.prob > 0:
            dropout_mask = torch.zeros(x.size(), device=x.device).uniform_()
            dropout_mask = dropout_mask >= self.prob
            return x * dropout_mask / (1-self.prob) if self.prob != 1 else x * dropout_mask
        else:
            return x


if __name__ == '__main__':
    a = Dropout(0.1)

    aa = torch.arange(1, 10, 1)
    print(aa)
    aa = a(aa)
    print(aa)
