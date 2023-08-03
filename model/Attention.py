import math
import torch
from torch import nn
from model.utils.Softmax import softmax
from model.utils.clone_layers import clones
from model.utils.Linear import Linear
from model.utils.Dropout import Dropout


def attention(q, k, v, mask=None, dropout=None):
    d_k = k.size(-1)
    res = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        res.masked_fill_(mask == 0, -torch.inf)
    res = softmax(res, -1)
    if dropout is not None:
        res = dropout(res)
    res = res.matmul(v)

    return res


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout_p=0.1) -> None:
        super().__init__()
        self.d_k = d_model // h
        self.h = h

        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.dropout = Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        nbatches = q.size(0)

        qw = self.w_q(q)
        kw = self.w_k(k)
        vw = self.w_v(v)

        q = qw.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        k = kw.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        v = vw.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x = attention(q, k, v, mask, self.dropout)
        x = x.transpose(1, 2).reshape(nbatches, -1, self.h*self.d_k)

        return self.w_o(x)


class MultiHeadAttention_SHORT(nn.Module):
    def __init__(self, d_model, h, dropout_p=0.1) -> None:
        super().__init__()
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(Linear(d_model, d_model), 4)
        self.dropout = Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        nbatches = q.size(0)

        qw, kw, vw = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (q, k, v))]

        x = attention(qw, kw, vw, mask, self.dropout).transpose(1, 2).reshape(nbatches, -1, self.h*self.d_k)

        return self.linears[3](x)


if __name__ == '__main__':
    rand_tensor = torch.randn([5, 10, 512])
    msa = MultiHeadAttention(512, 8)
    msa(rand_tensor, rand_tensor, rand_tensor)

    result = attention(rand_tensor, rand_tensor, rand_tensor)
    print(result.size())
