import torch
from torch import nn
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from utils.get_torch_device import get_torch_device
from Unicoder import Unicoder, UnicoderLayer
from Generator import Generator
from utils.FeedForward import FeedForward
from Attention import MultiHeadAttention
from EncoderDecoder_unicoder import EncoderDecoder_unicoder


def make_model(src_vocab, tgt_vocab, N: int, d_model: int, h: int, d_ff: int, dropout_p: float):
    encoder_sublayers = [MultiHeadAttention(d_model, h), FeedForward(d_model, d_ff)]
    decoder_sublayers = [MultiHeadAttention(d_model, h), MultiHeadAttention(d_model, h), FeedForward(d_model, d_ff)]

    encoder_layer = UnicoderLayer(d_model, encoder_sublayers, dropout_p=dropout_p)
    decoder_layer = UnicoderLayer(d_model, decoder_sublayers, dropout_p=dropout_p)

    encoder = Unicoder(encoder_layer, N)
    decoder = Unicoder(decoder_layer, N)

    generator = Generator(d_model, len(tgt_vocab))

    model = EncoderDecoder_unicoder(
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
    # with open('model.txt', 'wt') as f:
    #     f.write(str(model))
    # for m in model.parameters():
    #     print(m.device)
    inp = torch.arange(3,23,1).view(5,4).to(device=device)
    tgt = torch.arange(3,23,1).view(5,4).to(device=device)
    import time
    t = time.time()
    outp = model(inp, tgt)
    tt = time.time()-t
    print(round(tt,3))
    print(outp.shape)
    