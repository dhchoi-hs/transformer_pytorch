import torch
from torch import nn


class Dropout(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()

        if p < 0 or p > 1:
            print('[ERROR] Invalid value of dropout!')
            raise ValueError

        self.p = p
            
    def forward(self, x):
        if self.training and self.p > 0:
            dropout_mask = torch.zeros(x.size()).uniform_()
            dropout_mask = dropout_mask >= self.p
            return x * dropout_mask / (1-self.p) if self.p != 1 else x * dropout_mask
        else:
            return x


if __name__ == '__main__':
    a = Dropout(0.5)
    
    aa = torch.arange(1, 10, 1)
    print(aa)
    aa = a(aa)
    print(aa)
