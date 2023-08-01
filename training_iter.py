import time
from tqdm import tqdm
import torch


def run_step(a_data, model, criterion, optim=None, train_mode=True, device=None):
    if train_mode:
        assert optim, 'optimizer must be set in training mode.'
    x, y = a_data
    x = x.to(device)
    y = y.to(device)
    
    if train_mode:
        optim.zero_grad()

    pad_mask = (x != model.padding_idx).unsqueeze(-2).unsqueeze(-2)
    output = model(x, pad_mask)

    y_masked = y.nonzero(as_tuple=True)
    output_only_masked = output[y_masked]
    
    y_only_masked = y[y_masked]
    output_only_masked = torch.matmul(output_only_masked, model.emb.table.T)

    loss = criterion(output_only_masked, y_only_masked)

    if train_mode:
        loss.backward()
        model.emb.table.grad[model.emb.padding_idx] = torch.zeros_like(model.emb.table.grad[model.emb.padding_idx])
        optim.step()

    loss_item = loss.item()

    with torch.no_grad():
        output_labels = output_only_masked.argmax(dim=-1)
        a = torch.count_nonzero(output_labels == y_only_masked)
        acc = a.item() / y_only_masked.size(0)

    return loss_item, acc


def run_epoch(dataset, model, criterion, optim=None, train_mode=True, sleep=None, device=None):
    if train_mode:
        assert optim, 'optimizer must be set in training mode.'
    running_loss = 0.
    running_acc = 0.
    total_step = len(dataset)
    pbar = tqdm(dataset)
    mode = 'train' if train_mode else 'valid'
    for data in pbar:
        step_loss, step_acc = run_step(data, model, criterion, optim, train_mode, device)
        running_loss += step_loss
        running_acc += step_acc
        pbar.set_description(f'{mode} loss: {round(step_loss, 4):>8} acc: {round(step_acc, 4):>8}')
        if sleep:
            time.sleep(sleep)

    return running_loss/total_step, running_acc/total_step
