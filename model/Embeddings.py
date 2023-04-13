import torch
from torch import nn


class Embeddings(nn.Module):
    def __init__(self, vocab_len, d_model=512) -> None:
        super().__init__()
        
        self.table = nn.Parameter(torch.randn([vocab_len, d_model], requires_grad=True))
        self.d_model = d_model
    
    def forward(self, x):
        return self.table[x] * (self.d_model**(1/2))


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

    embs = Embeddings(len(bpe_dict))
    embs.to(device=device)
    res = embs(datas)

    print(res.shape)
    print(embs.state_dict())