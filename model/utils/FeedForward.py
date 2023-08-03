from torch import nn
from model.utils.Linear import Linear
from model.utils import activation_functions
from model.utils.Dropout import Dropout


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation, dropout_p) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.activation = getattr(activation_functions, activation)
        self.dropout = Dropout(dropout_p)

    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.w2(x)
