import torch
from torch import nn
from utils.Softmax import log_softmax
from utils.Linear import Linear


class Generator(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super().__init__()
        self.lin = Linear(d_model, vocab)
    
    def forward(self, x):
        return log_softmax(self.lin(x), dim=-1)


if __name__ == '__main__':
    inp = torch.randn([10,50,512])
    gen = Generator(512, 2000)
    r = gen(inp)
    print(r.size())
    r2 = torch.max(r, dim=-1)
    r3 = torch.argmax(r, dim=-1, keepdim=True)
    # print(r2.size())
    aa = r[r3]

    f = 1
