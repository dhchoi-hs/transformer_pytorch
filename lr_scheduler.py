

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
