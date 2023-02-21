from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, device=device)
        self.w2 = nn.Linear(d_ff, d_model, device=device)
        
    def forward(self, x):
        x = self.w1(x)
        x[x<0] = 0
        return self.w2(x)
