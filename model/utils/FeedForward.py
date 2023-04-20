from torch import nn
from model.utils.Linear import Linear


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = x.masked_fill(x < 0, 0)  # ReLU
        return self.w2(x)
