import sys
import os
import glob
import re
import argparse
import shutil
import json
import time
from datetime import datetime
import yaml
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset_loader.mlm_dataset import mlm_dataloader
from models.lm_encoder import lm_encoder
from model.utils.get_torch_device import get_torch_device


torch.manual_seed(7)


def run_epoch(dataset, model, criterion, optim=None, train_mode=True, sleep=None, device=None):
    if train_mode:
        assert optim, 'optimizer must be set in training mode.'
    running_loss = 0.
    total_step = len(dataset)
    pbar = tqdm(dataset)
    mode = 'train' if train_mode else 'valid'
    for data in pbar:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        
        if train_mode:
            optim.zero_grad()

        pad_mask = (x != model.padding_idx).unsqueeze(-2).unsqueeze(-2)
        output = model(x, pad_mask)

        y_masked = y.bool()
        output_only_masked = output[y_masked]
        
        y_only_masked = y[y_masked]
        output_only_masked = model.lin(output_only_masked)

        loss = criterion(output_only_masked, y_only_masked)

        if train_mode:
            loss.backward()
            optim.step()

        item = loss.item()
        running_loss += item

        output_labels = output_only_masked.argmax(dim=-1)
        a = torch.count_nonzero(output_labels == y_only_masked)
        acc = a.item() / y_only_masked.size(0)
        pbar.set_description(f'{mode} loss: {round(item, 4):>8} acc: {round(acc, 4):>8}')
        if sleep:
            time.sleep(sleep)

    return running_loss / total_step, acc


def save_model(model_dir, model_files, keep_last_models):
    model_file = os.path.join(model_dir, f'model_{i+1}.pt')
    torch.save(model.state_dict(), model_file)
    model_files.append(model_file)
    if len(model_files) > keep_last_models:
        for model_file in model_files[:-keep_last_models]:
            try:
                os.remove(model_file)
            except Exception as e:
                print(f'[WARNING] Deleting model file fails. {model_file}, {e}')
        model_files = model_files[-keep_last_models:]
    
    return model_files


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
    if not model_dir:
        model_dir = f'output/model_{datetime.now().strftime("%Y%m%d%H%M%S")}'

    keep_last_models = config['keep_last_models']
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
    vocab_file = config['vocab_file']
    vocab_start_token_id = config['vocab_start_token_id']
    train_dataset_files = config['train_dataset_files']
    valid_dataset_files = config['valid_dataset_files']

    assert keep_last_models > 0, 'keep_last_models config value must be greater than 0.'

    with open(vocab_file, 'rt') as f:
        vocab = json.load(f)

    if os.path.exists(model_dir):
        if not resume:
            print(f'[ERROR] model directory ({model_dir}) already exists.')
            sys.exit()
    else:
        if resume:
            print(f'[ERROR] model directory ({model_dir}) for resume not exists.')
            sys.exit()
        else:
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy(args.config, model_dir)

    train_dataloader = mlm_dataloader(train_dataset_files, vocab, vocab_start_token_id, seq_len, batch_size)
    valid_dataloader = mlm_dataloader(valid_dataset_files, vocab, vocab_start_token_id, seq_len, batch_size)
    device = get_torch_device(cuda_index)

    model = lm_encoder(
        d_model, h, ff, n_layers, len(vocab), padding_idx=vocab['__PAD__'],
        dropout_p=p_dropout, use_torch_module=False
    )
    model.to(device=device)

    separator = '='*80
    print(separator)
    num_params = 0
    for name, params in model.named_parameters():
        print(name, params.shape)
        if params.dim() > 1:
            torch.nn.init.xavier_uniform_(params)
        num_params += params.numel()
    print(
        f"{separator}\n"
        f"{model}\n"
        f"{separator}\n"
        f"Number of parameters: {num_params:,}\n"
        f"{separator}\n"
    )

    sw = SummaryWriter(os.path.join(model_dir, 'logs'))

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if resume:
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        def get_epoch_of_model_file(x):
            model_epoch = re.search(r'(?<=^model_)\d+(?=.pt$)', os.path.basename(x))
            return 0 if not model_epoch else int(model_epoch.group())
        
        model_files = sorted(glob.glob(os.path.join(model_dir, 'model_*.pt')), key=get_epoch_of_model_file)
        epoch = checkpoint['epoch']
    else:
        dataloader = train_dataloader or valid_dataloader
        sw.add_graph(model, dataloader.dataset[0][0].to(device))
        model_files = []
        epoch = 0

    sleep_between_step = .00
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    try:
        for i in range(epoch, epochs):
            t1 = time.time()
            training_sec_per_epoch = {}
            if train_dataloader is not None:
                training_started = time.time()
                model.train()
                train_loss, train_acc = run_epoch(train_dataloader, model, loss_fn, optim, True, sleep_between_step, device)
                training_sec_per_epoch['train'] = time.time() - training_started
                sw.add_scalar('Loss/train', train_loss, i+1)
                sw.add_scalar('Acc/train', train_acc, i+1)

            if valid_dataloader is not None:
                valid_started = time.time()
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = run_epoch(valid_dataloader, model, loss_fn, None, False, sleep_between_step, device)
                training_sec_per_epoch['valid'] = time.time() - valid_started
                sw.add_scalar('Loss/validation', val_loss, i+1)
                sw.add_scalar('Acc/validation', val_acc, i+1)

            sw.add_scalars('elapsed_sec_per_epoch', training_sec_per_epoch, i+1)
            model_files = save_model(model_dir, model_files, keep_last_models)
            print(f'epoch {i+1}. train_loss: {round(train_loss,4):>8}, train_acc: {round(train_acc, 4):>8}, val_loss: {round(val_loss,4):>8}, val_acc: {round(val_acc, 4):>8}, elapsed: {round(time.time()-t1,3)}s')
    except KeyboardInterrupt:
        print("[INFO] Training stopped.")
    except Exception as e:
        print(f'[ERROR] Exception occured during training. {e}')
    
    torch.save(
        {
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },
        os.path.join(model_dir, 'checkpoint.pt')
    )
    sw.close()
