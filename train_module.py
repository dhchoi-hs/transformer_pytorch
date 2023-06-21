import abc


class TrainModule(abc.ABC):
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device
        self.model.to(device=self.device)

    @abc.abstractmethod
    def configure_optimizer(self, config):
        optim = None
        scheduler = None
        return optim, scheduler

    @abc.abstractmethod
    def train_step(self, batch):
        loss = 0.
        return loss

    @abc.abstractmethod
    def validate_step(self, batch):
        pass

    def backward(self, loss, optim):
        loss.backward()
