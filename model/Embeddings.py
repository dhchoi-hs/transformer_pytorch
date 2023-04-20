import torch
from torch import nn
from math import sqrt


class Embeddings(nn.Module):
    def __init__(self, vocab_len, d_model=512, padding_idx=0) -> None:
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        self.table = nn.Parameter(torch.randn([vocab_len-1, d_model]), requires_grad=True)
        # self.padding_emb = nn.Parameter(torch.zeros([1, self.d_model]), requires_grad=False)
        self.padding_emb_to_zero()

    def padding_emb_to_zero(self):
        self.table.data[self.padding_idx] = torch.zeros(self.d_model)

    def padding_emb_grad_to_zero(self):
        self.table[self.padding_idx].grad = torch.zeros(self.d_model)

    def forward(self, x):
        if self.table[self.padding_idx].data.any().item() is True:
            self.padding_emb_to_zero()
        # table = torch.cat([self.table[:self.padding_idx], self.padding_emb, self.table[self.padding_idx:]])
        return self.table[x] * sqrt(self.d_model)


if __name__ == '__main__':
    import json
    from utils.get_torch_device import get_torch_device

    device = get_torch_device()

    with open('bpe_processing/BPE_dict.json', 'r') as f:
        bpe_dict = json.load(f)

    with open('/Users/dhchoi/Downloads/dict/tmp_bpe_files/ko_bpe_emb_ID.txt', 'r') as f:
        lines = f.readlines()

    print(f'length of bpd dict: {len(bpe_dict)}')
    print(f'length of text lines: {len(lines)}')
    
    data = [list(map(int, t.replace('\n', '').split(','))) for t in lines]
    max_len = len(max(data, key=lambda x: len(x)))
    pad_index = bpe_dict['__PAD__']
    data = [l+([pad_index]*(max_len-len(l))) for l in data]
    datas = torch.LongTensor(data[:20]).to(device=device)

    embs = Embeddings(len(bpe_dict), padding_idx=pad_index)
    embs.to(device=device)
    res = embs(datas)
    res = embs(datas)

    print(res.shape)
    print(embs.state_dict())