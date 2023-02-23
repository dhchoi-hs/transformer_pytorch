from torch import nn
from Encoder import Encoder
from Decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        return self.decoder(self.encoder(x))