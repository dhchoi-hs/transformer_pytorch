from torch import nn
from copy import deepcopy


def clones(layer, n):
    return nn.ModuleList([deepcopy(layer) for _ in range(n)])
