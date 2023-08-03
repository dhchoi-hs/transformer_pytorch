from torch import nn
from model.Embeddings import Embeddings
from model.PositionalEncoding import PositionalEncoding
from model.Encoder import Encoder, EncoderLayer


class lm_encoder(nn.Module):
    def __init__(self, d_model, h, ff, n_layers, n_vocabs, padding_idx=0,
                 dropout_p=0.1, activation='relu'):
        super().__init__()
        self.padding_idx = padding_idx
        self.pe = PositionalEncoding(n_vocabs, d_model, 0)
        self.emb = Embeddings(n_vocabs, d_model, padding_idx)
        self.encoder = Encoder(EncoderLayer(d_model, h, ff, dropout_p,
                                            activation=activation), n_layers)
        # self.l = nn.Linear(d_model, n_vocabs)

    def forward(self, x, src_mask=None,):
        x = self.emb(x)
        x = self.pe(x)
        x = self.encoder(x, src_mask)

        return x


class lm_encoder_torch(nn.Module):
    def __init__(self, d_model, h, ff, n_layers, n_vocabs, padding_idx=0,
                 dropout_p=0.1, activation='relu'):
        super().__init__()
        self.padding_idx = padding_idx
        self.emb = nn.Embedding(n_vocabs, d_model, padding_idx)
        self.pe = PositionalEncoding(n_vocabs, d_model, dropout_p)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, h, ff, dropout_p, activation=activation, batch_first=True,
            norm_first=True), n_layers, norm=nn.LayerNorm([d_model]))
        self.l = nn.Linear(d_model, n_vocabs)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.emb(x)
        x = self.pe(x)
        x = self.encoder(x, src_mask, src_key_padding_mask=src_key_padding_mask)

        return x
