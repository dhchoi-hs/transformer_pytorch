import torch
from torch import nn
from utils.SublayerConnection import SublayerConnection
from utils.clone_layers import clones
from model.FeedForward import FeedForward
from Attention import MultiHeadAttention
from utils.LayerNorm import LayerNorm


class PassX:
    pass


class UnicoderLayer(nn.Module):
    def __init__(self, d_model: int, sublayers: list, dropout_p=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.sublayer_connections = clones(SublayerConnection(d_model, dropout_p), len(sublayers))
        self.sublayers = nn.ModuleList(sublayers)
    
    def forward(self, x, params):
        for sublayer_connection, sublayer, param in zip(self.sublayer_connections, self.sublayers, params):
            x = sublayer_connection(x, lambda _x: sublayer(*[_x if isinstance(_param, PassX) else _param for _param in param]))

        return x


class Unicoder(nn.Module):
    def __init__(self, layer, n_encoder_layers: int) -> None:
        super().__init__()
        self.layers = clones(layer, n_encoder_layers)
        self.layer_norm = LayerNorm(layer.d_model)
    
    def forward(self, x, params):
        for layer in self.layers:
            x = layer(x, params)

        return self.layer_norm(x)
    

if __name__ == '__main__':
    sublayers = [MultiHeadAttention(512, 8), MultiHeadAttention(512, 8), FeedForward(512, 2048)]
    unicoder_layer = UnicoderLayer(512, sublayers)
    mem_t = torch.randn([10, 512])
    params = [(PassX(),)*3, (PassX(), mem_t, mem_t), (PassX(),)]
    x = torch.randn([10,512])
    outp = unicoder_layer(x, params)
    print(outp)

    ins = Unicoder(unicoder_layer, 6)
    outp2 = ins(x, params)
    print(outp2)
