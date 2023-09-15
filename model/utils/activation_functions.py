import math
from typing import Callable
import torch
from torch import nn



# MAX_EXPONENT_VAL = torch.trunc(torch.log(torch.nan_to_num(torch.tensor(float('inf'))))*10000)/10000


# def relu(x):
#     return x.masked_fill(x < 0, 0)


# def gelu(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)*(x + 0.044715 * x ** 3)))


# def silu(x):
#     # prevent torch.exp(-x) getting infinity value.
#     x = torch.where(x > -MAX_EXPONENT_VAL, x, -MAX_EXPONENT_VAL)
#     return x * 1 / (1 + torch.exp(-x))


def relu():
    return nn.ReLU()


def gelu():
    return nn.GELU()


def silu():
    return nn.SiLU()


swish: Callable = silu


def _test():
    inp = torch.range(-5, 5, 0.1)

    for activation in ['relu', 'gelu', 'silu']:
        output = globals()[activation](inp)
        plt.plot(inp.tolist(), output.tolist(), label=activation)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    _test()
