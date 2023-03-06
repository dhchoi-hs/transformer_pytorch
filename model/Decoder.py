from torch import nn
from utils.SublayerConnection import SublayerConnection
from utils.clone_layers import clones
from utils.FeedForward import FeedForward
from Attention import MultiHeadAttention
from utils.LayerNorm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout_p=0.1, device=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.sublayer_connections = clones(SublayerConnection(d_model, dropout_p), 3)
        self.self_mha = MultiHeadAttention(d_model, h, device)
        self.mha = MultiHeadAttention(d_model, h, device)
        self.ff = FeedForward(d_model, d_ff, device)
    
    def forward(self, x, m, src_mask=None, tgt_mask=None):
        x = self.sublayer_connections[0](x, lambda _x: self.self_mha(_x, _x, _x, mask=tgt_mask))
        x = self.sublayer_connections[1](x, lambda _x: self.mha(_x, m, m, mask=src_mask))

        return self.sublayer_connections[2](x, self.ff)


class Decoder(nn.Module):
    def __init__(self, layer, n_encoder_layers: int) -> None:
        super().__init__()
        self.layers = clones(layer, n_encoder_layers)
        self.layer_norm = LayerNorm(layer.d_model)
    
    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, mem, src_mask, tgt_mask)

        return self.layer_norm(x)
