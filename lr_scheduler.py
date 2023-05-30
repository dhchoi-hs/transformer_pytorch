import math
from torch.optim import lr_scheduler


SCHEDULERS = [
    None,
    'WarmupExp',
    'CosineAnnealingWarmRestarts',
    'CustomCosineAnnealingWarmRestarts'
]


def create_lr_lambda(gamma, steps_per_epoch):
    """
    Create learning rate schedule lambda for torch LambdaLR().
    warm up 1 epoch, lr changed every step.
    after warm up, lr decreases exponentially.
    Example:
        >>> scheduler = lr_scheduler.LambdaLR(optim, create_lr_lambda(lr_gamma, len(train_dataloader)), step)
    """
    def lr_lambda(steps):
        epoch = steps / steps_per_epoch
        if epoch < 1.:
            return (steps+1) / steps_per_epoch

        return gamma ** (int(epoch)-1)

    return lr_lambda


class CustomCosineAnnealingWarmRestarts(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        self.max_multply = 1
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur <= self.T_up:
            return [(self.T_cur) / self.T_up * (base_lr*self.max_multply) for base_lr in self.base_lrs]
        else:
            return [(base_lr * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2)*self.max_multply
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
                self.max_multply = self.gamma**self.cycle
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    self.max_multply = self.gamma**self.cycle
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def create_lr_scheduler(optim, scheduler, steps_per_epoch, **kwargs):
    if not scheduler:
        scheduler = None
    elif scheduler == SCHEDULERS[1]:
        scheduler = lr_scheduler.LambdaLR(optim, create_lr_lambda(kwargs[0], steps_per_epoch=steps_per_epoch))
    elif scheduler == SCHEDULERS[2]:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optim, **kwargs)
    elif scheduler == SCHEDULERS[3]:
        scheduler = CustomCosineAnnealingWarmRestarts(optim, **kwargs)
    else:
        raise ValueError(f'Unknown scheduler name {scheduler}')

    return scheduler


def test():
    lin = Linear(2, 2)
    optim = Adam(lin.parameters(), lr=5e-4)
    kwargs = {
        'T_0': 5,
        'T_mult': 2,
        'T_up': 1,
        'gamma': 0.5
    }
    scheduler = create_lr_scheduler(optim, 'CustomCosineAnnealingWarmRestarts', 1, **kwargs)

    epoch = 0.
    epochs = []
    lrs = []

    while epoch <= 20:
        lr = scheduler.get_lr()
        epochs.append(epoch)
        lr = round(lr[0], 10)
        lrs.append(lr)
        print(epoch, lr)
        epoch += 0.1
        epoch = round(epoch, 2)
        scheduler.step(epoch)

    plt.plot(epochs, lrs)
    plt.title('learning rate graph')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    from torch.optim import Adam
    from torch.nn import Linear
    import matplotlib.pyplot as plt

    test()
