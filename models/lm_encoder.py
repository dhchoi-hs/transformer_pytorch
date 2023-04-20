import torch
from torch import nn
from model.Embeddings import Embeddings
from model.PositionalEncoding import PositionalEncoding
from model.Encoder import Encoder, EncoderLayer
from model.utils.Linear import Linear
from model.utils.Softmax import Softmax


class lm_encoder(nn.Module):
    def __init__(self, d_model, h, ff, n_layers, n_vocabs, padding_idx=0, dropout_p=0.1, use_torch_module=False):
        super().__init__()
        self.padding_idx = padding_idx
        self.pe = PositionalEncoding(n_vocabs, d_model, dropout_p)
        if not use_torch_module:
            self.emb = Embeddings(n_vocabs, d_model, padding_idx)
            self.encoder = Encoder(EncoderLayer(d_model, h, ff, dropout_p), n_layers)
            self.lin = Linear(d_model, n_vocabs)
        else:
            self.emb = nn.Embedding(n_vocabs, d_model, padding_idx)
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, h, ff, dropout_p, batch_first=True), n_layers, enable_nested_tensor=False)
            self.lin = nn.Linear(d_model, n_vocabs)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, src_mask=None,):
        x = self.emb(x)
        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=src_mask)

        return x
        # x = self.emb(x)
        x = self.lin(x)
        return x
        return self.softmax(x)