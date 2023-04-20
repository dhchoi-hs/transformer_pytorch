import os
import json
import time
import torch
from tqdm import tqdm
from utils.get_torch_device import get_torch_device
from torch.utils.tensorboard import SummaryWriter
from dataset_loader.mlm_dataset import mlm_dataloader
from models.lm_encoder import lm_encoder


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
        outputs = model(x, pad_mask=pad_mask)
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
        pbar.set_description(f'loss: {round(item, 4):>8} acc: {round(acc, 4):>8}')
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
        # outputs = model(x, pad_mask=pad_mask)
        outputs = model(x)
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
        pbar.set_description(f'loss: {round(item, 4):>8} acc: {round(acc, 4):>8}')
        if sleep:
            time.sleep(sleep)

    return running_loss / total_step, acc

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda_index', type=str, default=0)
    arg = ap.parse_args()

    d_model = 512
    h = 8
    ff = 2048
    n_layers = 3
    batch_size = 128
    p_dropout = 0.
    seq_len = 256

    vocab_dir = '/home/dhchoi/projects/transformer_pytorch/kypark_dataset/bpe_dict'
    vocab_dict = ['BPE_char_dict.json', 'BPE_byte_dict.json']
    train_dataset_dir = '/home/dhchoi/projects/transformer_pytorch/kypark_dataset/encoded_corpus_files'
    valid_dataset_dir = '/home/dhchoi/projects/pytorch_txfmr/data/text/AI_Hub_KoEn/encoded'
    log_dir = None
    train_dataset_files = [
        'BPE_char_en_tr_090.txt',
        'BPE_char_ko_tr_090.txt',
        'BPE_byte_en_tr_090.txt',
        'BPE_byte_ko_tr_090.txt',
    ]
    valid_dataset_files = [
        'BPE_char_en_va_010.txt',
        'BPE_char_ko_va_010.txt',
        'BPE_byte_en_va_010.txt',
        'BPE_byte_ko_va_010.txt',
    ]

    device = get_torch_device(arg.cuda_index)
    with open(os.path.join(vocab_dir, vocab_dict[0]), 'rt') as f:
        vocab = json.load(f)

    train_dataloader = mlm_dataloader(
        [
           os.path.join(train_dataset_dir, train_dataset_files[0]),
           os.path.join(train_dataset_dir, train_dataset_files[1])
        ],
        vocab, 6, seq_len, batch_size
    )
    valid_dataloader = mlm_dataloader(
        [
            os.path.join(train_dataset_dir, valid_dataset_files[0]),
            os.path.join(train_dataset_dir, valid_dataset_files[1])
            ],
        vocab, 6, seq_len, batch_size
    )

    model = lm_encoder(d_model, h, ff, n_layers, len(vocab), padding_idx=vocab['__PAD__'], dropout_p=p_dropout)
    model.to(device=device)

    num_params = 0
    for name, params in model.named_parameters():
        print(name, params.shape)
        if params.dim() > 1:
            torch.nn.init.xavier_uniform_(params)
        num_params += params.numel()

    print(f'Number of parameters: {num_params:,} ')

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    sw = SummaryWriter(log_dir)
    sw.add_graph(model, train_dataloader[0][0][0].to(device))

    model.train()

    sleep = .05
    try:
        for i in range(1000):
            t1 = time.time()
            if not model.training:
                model.train()
            train_loss, train_acc = train_epoch(train_dataloader, model, loss_fn, optim, sleep, device)
            sw.add_scalar('Loss/train', train_loss, i+1)
            sw.add_scalar('Acc/train', train_acc, i+1)

            if valid_dataloader.dataset is not None:
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = valid_epoch(valid_dataloader, model, loss_fn, sleep, device)
                sw.add_scalar('Loss/validation', val_loss, i+1)
                sw.add_scalar('Acc/validation', val_acc, i+1)
            print(f'epoch {i+1}. train_loss: {round(train_loss,4):>8}, train_acc: {round(train_acc, 4):>8}, val_loss: {round(val_loss,4):>8}, val_acc: {round(val_acc, 4):>8}, elapsed: {round(time.time()-t1,3)}s')
    except KeyboardInterrupt:
        print("Stop")
    
    if model:
        torch.save(model.state_dict(), 'model.pt')