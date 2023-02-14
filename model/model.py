import torch
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from utils.get_torch_device import get_torch_device


device = get_torch_device()
vocab_len = 20000
d_model = 512

d = torch.tensor([[1,2,3,4,],[5,6,7,8]], dtype=torch.long, device=device)
embeddings = Embeddings(vocab_len, d_model, device)
pe = PositionalEncoding(vocab_len, d_model, device=device)

res = pe(embeddings(d))

print(res.shape)
