from collections import defaultdict
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

    # pad_mask = (x == model.padding_idx)
    # output = model(x, src_key_padding_mask=pad_mask)
    pad_mask = (x != model.padding_idx).unsqueeze(-2).unsqueeze(-2)
    output = model(x, pad_mask)

    y_predict_token_pos = y != -1
    y_only_masked = y[y_predict_token_pos]

    output_only_masked = output[y_predict_token_pos]
    output_only_masked = torch.matmul(output_only_masked, model.emb.table.T)
    # output_only_masked = torch.matmul(output_only_masked, model.emb.weight.T)
    # output_only_masked = model.l(output_only_masked)

    loss = criterion(output_only_masked, y_only_masked)

    if train_mode:
        loss.backward()
        model.emb.table.grad[model.emb.padding_idx] = \
            torch.zeros_like(model.emb.table.grad[model.emb.padding_idx])
        optim.step()

    with torch.no_grad():
        output_labels = output_only_masked.argmax(dim=-1)
        a = torch.count_nonzero(output_labels == y_only_masked)
        acc = a.item() / y_only_masked.size(0)

    metrics = {
        'loss': loss.item(),
        'acc': acc
    }
    return metrics


def run_step_fine_tuning(a_data, model, criterion, optim=None, train_mode=True, device=None):
    if train_mode:
        assert optim, 'optimizer must be set in training mode.'
    x, y = a_data
    x = x.to(device)
    y = y.to(device)

    if train_mode:
        optim.zero_grad()

    pad_mask = (x != model.pretrained_model.padding_idx).unsqueeze(-2).unsqueeze(-2)
    output = model(x, pad_mask)

    loss = criterion(output, y)

    if train_mode:
        loss.backward()
        optim.step()

    with torch.no_grad():
        output_labels = output >= 0.5
        a = torch.count_nonzero(output_labels == y)
        acc = a.item() / y.size(0)
        tp = torch.count_nonzero(y[output_labels==1]).item()
        fp = torch.count_nonzero(y[output_labels==1] == 0).item()
        fn = torch.count_nonzero(y[output_labels==0]).item()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2*(precision*recall)/(precision+recall)

    metrics = {
        'loss': loss.item(),
        'acc': acc,
        'f1': f1
    }
    return metrics


def run_epoch(
        dataset,
        model,
        criterion,
        optim=None,
        train_mode=True,
        device=None,
        fine_tuning=False):
    if train_mode:
        assert optim, 'optimizer must be set in training mode.'

    total_step = len(dataset)
    pbar = tqdm(dataset)
    mode = 'train' if train_mode else 'valid'
    epoch_metrics = defaultdict(float)
    run = run_step if not fine_tuning else run_step_fine_tuning
    for data in pbar:
        metrics = run(data, model, criterion, optim, train_mode, device)
        pbar.set_description(f'{mode} loss: {round(metrics["loss"], 4):>8} acc: {round(metrics["acc"], 4):>8}')
        for metric, score in metrics.items():
            epoch_metrics[metric] += score

    metrics_avg = {}
    for metric, score in epoch_metrics.items():
        metrics_avg[metric] = score/total_step

    return metrics_avg
