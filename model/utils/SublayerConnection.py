from torch import nn
from utils.Dropout import Dropout
from utils.LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p) -> None:
        super().__init__()
        self.layer_normalizing = LayerNorm(size)
        self.dropout = Dropout(dropout_p)
    
    def forward(self, x, sublayer):
        # residual connection after sublayer
        return x + self.dropout(sublayer(self.layer_normalizing(x)))
