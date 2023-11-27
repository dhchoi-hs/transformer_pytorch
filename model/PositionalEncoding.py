import torch
from torch import nn
import os
from model.utils.Dropout import Dropout


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_len, d_model=512, dropout=0.1) -> None:
        super().__init__()

        self.pe = torch.zeros([vocab_len, d_model])

        pe = torch.arange(0, vocab_len).type(torch.float32).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).type(torch.float32)
        data = pe/(10000**(_2i/d_model))
        
        self.pe[:, ::2] = torch.sin(data)
        self.pe[:, 1::2] = torch.cos(data)

        self.dropout = Dropout(dropout)
        self.register_buffer('pe1', self.pe)
    
    def forward(self, x):
        return self.dropout(x + self.pe1.to(x.device)[:x.size(-2), ::])


if __name__ == '__main__':
    from utils.get_torch_device import get_torch_device
    import matplotlib.pyplot as plt

    device = get_torch_device()
    pe = PositionalEncoding(32, 512, 0.1)

    x = pe(torch.randn([3,12,512]))
    print(x)
    print(pe.state_dict())

    # data = pe.pe.cpu().detach().numpy()

    # n = plt.matshow(data)
    # plt.colorbar(n)
    # plt.show()
