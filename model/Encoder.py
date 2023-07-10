from torch import nn
from model.utils.SublayerConnection import SublayerConnection
from model.utils.clone_layers import clones
from model.utils.FeedForward import FeedForward
from model.Attention import MultiHeadAttention
from model.utils.LayerNorm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout_p: float = 0.1,
                 activation='relu') -> None:
        super().__init__()
        self.d_model = d_model
        self.sublayer_connections = clones(
            SublayerConnection(d_model, dropout_p=dropout_p), 2)
        self.mha = MultiHeadAttention(d_model, h)
        self.ff = FeedForward(d_model, d_ff, activation)
    
    def forward(self, x, mask=None):
        x = self.sublayer_connections[0](x, lambda _x: self.mha(_x, _x, _x, mask))
        return self.sublayer_connections[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, n_encoder_layers: int) -> None:
        super().__init__()
        self.layers = clones(layer, n_encoder_layers)
        self.layer_norm = LayerNorm(layer.d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)
