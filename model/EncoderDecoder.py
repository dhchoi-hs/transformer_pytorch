from torch import nn
# from Encoder import Encoder
# from Decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, src_embed, tgt_embed, generator) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, x, tgt, src_mask=None, tgt_mask=None):
        return self.decode(tgt, self.encode(x, src_mask), src_mask=src_mask, tgt_mask=tgt_mask)

    def encode(self, x, src_mask=None):
        return self.encoder(self.src_embed(x), src_mask)

    def decode(self, x, m, src_mask=None, tgt_mask=None):
        return self.decoder(self.tgt_embed(x), m, src_mask, tgt_mask)
