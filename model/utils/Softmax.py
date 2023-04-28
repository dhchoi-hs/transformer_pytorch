import torch


def softmax(x, dim=None):
    dim = -1 if dim is None else dim
    return x.softmax(dim)
    # this function has problem divide by zero on backward.
    xe = torch.exp(x - abs(x.max(dim,keepdim=True)[0]))
    s = torch.sum(xe, dim=dim, keepdim=True)
    sf = xe / s

    return sf


def log_softmax(x, dim=None):
    dim = -1 if dim is None else dim
    x_off = x-abs(x.max(dim,keepdim=True)[0])
    return x_off - torch.log(torch.sum(torch.exp(x_off), dim=dim, keepdim=True))


class Softmax:
    def __init__(self, dim=None) -> None:
        self.dim = dim

    def __call__(self, x):
        dim = -1 if self.dim is None else self.dim
        return softmax(x, dim)

class LogSoftmax:
    def __init__(self, dim=None) -> None:
        super().__init__(dim)
    
    def __call__(self, x):
        dim = -1 if self.dim is None else self.dim
        return log_softmax(x, dim)
