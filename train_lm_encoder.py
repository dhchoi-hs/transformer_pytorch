import sys
import os
import argparse
import shutil
import json
import time
import yaml
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset_loader.mlm_dataset import mlm_dataloader
from models.lm_encoder import lm_encoder
from model.utils.get_torch_device import get_torch_device


torch.manual_seed(7)


def train_epoch(dataset, model, criterion, optim, sleep=None, device=None):
    running_loss = 0.
    total_step = len(dataset)
    pbar = tqdm(dataset)
    for data in pbar:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        
        optim.zero_grad()

        pad_mask = (x == model.padding_idx)
        outputs = model(x, pad_mask)
        y_masked = y.bool()
        outputs_only_masked = outputs[y_masked]
        y_only_masked = y[y_masked]
        outputs_only_masked = model.lin(outputs_only_masked)

        loss = criterion(outputs_only_masked, y_only_masked)
        
        loss.backward()

        optim.step()
        item = loss.item()
        running_loss += item

        output_labels = outputs_only_masked.argmax(dim=-1)
        a = torch.count_nonzero(output_labels == y_only_masked)
        acc = a.item() / y_only_masked.size(0)
        pbar.set_description(f'train loss: {round(item, 4):>8} acc: {round(acc, 4):>8}')
        if sleep:
            time.sleep(sleep)

    return running_loss / total_step, acc


def valid_epoch(dataset, model, criterion, sleep=None, device=None):
    running_loss = 0.
    total_step = len(dataset)
    pbar = tqdm(dataset)
    for data in pbar:
        x, y = data
        x = x.to(device)
        y = y.to(device)

        pad_mask = (x == model.padding_idx)
        outputs = model(x, pad_mask=pad_mask)
        y_masked = y.bool()
        outputs_only_masked = outputs[y_masked]
        y_only_masked = y[y_masked]
        outputs_only_masked = model.lin(outputs_only_masked)

        loss = criterion(outputs_only_masked, y_only_masked)
        
        item = loss.item()
        running_loss += item

        output_labels = outputs_only_masked.argmax(dim=-1)
        a = torch.count_nonzero(output_labels == y_only_masked)
        acc = a.item() / y_only_masked.size(0)
        pbar.set_description(f'valid loss: {round(item, 4):>8} acc: {round(acc, 4):>8}')
        if sleep:
            time.sleep(sleep)

    return running_loss / total_step, acc

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', type=str, default='configs/config_ln_encoder.yaml')
    ap.add_argument('-r', '--resume', default=False, action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f'[INFO] config file {args.config} not exists.')
        sys.exit()

    with open(args.config, 'rt') as config_file:
        config = yaml.load(config_file, yaml.SafeLoader)

    resume = args.resume

    model_dir = config['model_dir']
    cuda_index = config['cuda_index']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    d_model = config['d_model']
    h = config['h']
    ff = config['ff']
    n_layers = config['n_layers']
    p_dropout = config['p_dropout']
    seq_len = config['seq_len']
    vocab_start_token_id = config['vocab_start_token_id']
    vocab_file = config['vocab_file']
    train_dataset_files = config['train_dataset']['files']
    valid_dataset_files = config['valid_dataset']['files']

    with open(vocab_file, 'rt') as f:
        vocab = json.load(f)

    if os.path.exists(model_dir):
        if not resume:
            print(f'[INFO] model directory ({model_dir}) already exists.')
            sys.exit()
    else:
        if resume:
            print(f'[INFO] model directory ({model_dir}) for resume not exists.')
            sys.exit()
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(args.config, model_dir)
    

    train_dataloader = mlm_dataloader(train_dataset_files, vocab, vocab_start_token_id, seq_len, batch_size)
    valid_dataloader = mlm_dataloader(valid_dataset_files, vocab, vocab_start_token_id, seq_len, batch_size)

    device = get_torch_device(cuda_index)

    model = lm_encoder(d_model, h, ff, n_layers, len(vocab), padding_idx=vocab['__PAD__'], dropout_p=p_dropout, use_torch_module=True)
    model.to(device=device)
    if resume:
        checkpoint = torch.load(os.path.join(model_dir, 'model_checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])

    separator = '='*80
    print(separator)
    num_params = 0
    for name, params in model.named_parameters():
        print(name, params.shape)
        if params.dim() > 1:
            torch.nn.init.xavier_uniform_(params)
        num_params += params.numel()
    print(separator)
    print(model)
    print(separator)
    print(f'Number of parameters: {num_params:,} ')
    print(separator)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if resume:
        optim.load_state_dict(checkpoint('optimizer_state_dict'))

    sw = SummaryWriter(os.path.join(model_dir, 'logs'))
    dataloader = train_dataloader or valid_dataloader
    sw.add_graph(model, dataloader.dataset[0][0].to(device))

    model.train()

    sleep_per_step = .01
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    epoch = 0 if not resume else checkpoint['epoch']
    try:
        for i in range(epoch, epochs):
            t1 = time.time()
            if train_dataloader is not None:
                if not model.training:
                    model.train()
                train_loss, train_acc = train_epoch(train_dataloader, model, loss_fn, optim, sleep_per_step, device)
                sw.add_scalar('Loss/train', train_loss, i+1)
                sw.add_scalar('Acc/train', train_acc, i+1)

            if valid_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = valid_epoch(valid_dataloader, model, loss_fn, sleep_per_step, device)
                sw.add_scalar('Loss/validation', val_loss, i+1)
                sw.add_scalar('Acc/validation', val_acc, i+1)
            print(f'epoch {i+1}. train_loss: {round(train_loss,4):>8}, train_acc: {round(train_acc, 4):>8}, val_loss: {round(val_loss,4):>8}, val_acc: {round(val_acc, 4):>8}, elapsed: {round(time.time()-t1,3)}s')
    except KeyboardInterrupt:
        print("Training stopped.")
    
    torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch{i}.pt'))
    torch.save(
        {
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },
        os.path.join(model_dir, 'model_checkpoint.pt')
    )
