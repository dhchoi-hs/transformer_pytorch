import math
from typing import Callable
import torch


def relu(x):
    return x.masked_fill(x < 0, 0)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)*(x + 0.044715 * x ** 3)))


def silu(x):
    return x * 1 / (1 + torch.exp(-x))


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
