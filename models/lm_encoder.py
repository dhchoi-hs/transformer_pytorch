import torch
from torch import nn
from model.Embeddings import Embeddings
from model.PositionalEncoding import PositionalEncoding
from model.Encoder import Encoder, EncoderLayer


class lm_encoder(nn.Module):
    def __init__(self, d_model, h, ff, n_layers, n_vocabs, padding_idx=0, dropout_p=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.pe = PositionalEncoding(n_vocabs, d_model, dropout_p)
        self.emb = Embeddings(n_vocabs, d_model, padding_idx)
        self.encoder = Encoder(EncoderLayer(d_model, h, ff, dropout_p), n_layers)
    
    def forward(self, x, src_mask=None,):
        x = self.emb(x)
        x = self.pe(x)
        x = self.encoder(x, src_mask)

        return x