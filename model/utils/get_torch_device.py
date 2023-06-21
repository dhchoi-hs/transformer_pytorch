import torch


def get_torch_device(cuda_idx=None):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available() and cuda_idx is not None:
        cuda = f'cuda:{cuda_idx}'
        device = torch.device(cuda)
    else:
        device = torch.device('cpu')

    return device

