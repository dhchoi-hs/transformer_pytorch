import torch


def softmax(x, dim=None):
    dim = -1 if dim is None else dim
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def log_softmax(x, dim=None):
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


class Softmax:
    def __init__(self, dim=None) -> None:
        self.dim = dim

    def __call__(self, x):
        dim = -1 if self.dim is None else self.dim
        return softmax(x, dim)

class LogSoftmax(Softmax):
    def __init__(self, dim=None) -> None:
        super().__init__(dim)
    
    def __call__(self, x):
        dim = -1 if self.dim is None else self.dim
        return log_softmax(x, dim)
