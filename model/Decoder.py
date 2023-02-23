from torch import nn
from utils.SublayerConnection import SublayerConnection
from utils.clone_layers import clones
from utils.FeedForward import FeedForward
from model.Attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, device=None) -> None:
        super().__init__()
        self.sublayer_connections = clones(SublayerConnection(), 3)
        self.self_mha = MultiHeadAttention(d_model, h, device)
        self.mha = MultiHeadAttention(d_model, h, device)
        self.ff = FeedForward(d_model, d_ff, device)
    
    def forward(self, x, m):
        x = self.sublayer_connections[0](x, lambda _x: self.self_mha(_x, _x, _x))
        x = self.sublayer_connections[1](x, lambda _x: self.mha(_x, m, m))

        return self.sublayer_connections[2](x, self.ff)


class Decoder(nn.Module):
    def __init__(self, encoder_layer) -> None:
        super().__init__()
        self.layers = clones(encoder_layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
