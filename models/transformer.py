import torch
from torch import nn
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from utils.get_torch_device import get_torch_device
from EncoderDecoder import EncoderDecoder
from Encoder import EncoderLayer, Encoder
from Decoder import DecoderLayer, Decoder
from Generator import Generator


def make_model(src_vocab, tgt_vocab, N: int, d_model: int, h: int, d_ff: int, dropout_p: float):
    encoder_layer = EncoderLayer(d_model, h, d_ff, dropout_p=dropout_p)
    decoder_layer = DecoderLayer(d_model, h, d_ff)
    encoder = Encoder(encoder_layer, N)
    decoder = Decoder(decoder_layer, N)
    generator = Generator(d_model, len(tgt_vocab))

    model = EncoderDecoder(
        encoder,
        decoder,
        nn.Sequential(
            Embeddings(len(src_vocab), d_model),
            PositionalEncoding(len(src_vocab), d_model, dropout_p)
        ),
        nn.Sequential(
            Embeddings(len(tgt_vocab), d_model),
            PositionalEncoding(len(tgt_vocab), d_model, dropout_p)
        ),
        generator
    )

    return model


if __name__ == '__main__':
    import json
    with open('bpe_processing/BPE_dict.json', 'rt') as f:
        vocab = json.load(f)
    model = make_model(vocab, vocab, 6, 512, 8, 2048, 0.1)
    # model.cuda()
    device = get_torch_device()
    model.to(device=device)
    model.eval()

    # with open('model.txt', 'wt') as f:
    #     f.write(str(model))
    # for m in model.parameters():
    #     print(m.device)
    inp = torch.arange(3,23,1).view(5,4)
    tgt = torch.arange(3,23,1).view(5,4)
    outp = model(inp, tgt)
    print(outp)
    print(outp.shape)
