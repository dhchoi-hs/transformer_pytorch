import torch
from torch import nn
import os


class Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass


if __name__ == '__main__':
    import json
    with open('data_preprocessing/BPE_dict.json', 'r') as f:
        bpe_dict = json.load(f)
    print(f'length of bpd dict: {len(bpe_dict)}')

    embs = Embeddings()
    embs(1)
