from torch import nn
from utils.SublayerConnection import SublayerConnection
from utils.clone_layers import clones
from utils.FeedForward import FeedForward
from Attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, size, d_model, h, d_ff, device=None, dropout_p=0.1) -> None:
        super().__init__()
        
        self.sublayer_connections = clones(SublayerConnection(size, dropout_p=dropout_p), 2)
        self.mha = MultiHeadAttention(d_model, h, device)
        self.ff = FeedForward(d_model, d_ff, device)
    
    def forward(self, x, mask):
        x = self.sublayer_connections[0](x, lambda _x: self.mha(_x, _x, _x, mask))
        return self.sublayer_connections[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, encoder_layer) -> None:
        super().__init__()
        self.layers = clones(encoder_layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
