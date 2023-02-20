import torch


class Softmax:
    def __init__(self, dim=None) -> None:
        self.dim = dim

    def __call__(self, x):
        dim = -1 if self.dim is None else self.dim
        return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


class LogSoftmax(Softmax):
    def __init__(self, dim=None) -> None:
        super().__init__(dim)
    
    def __call__(self, x):
        return torch.log(super().__call__(x))
