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
from logger import init_logger, get_logger


torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


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

    y_masked = y.bool()
    output_only_masked = output[y_masked]
    
    y_only_masked = y[y_masked]
    output_only_masked = torch.matmul(output_only_masked, model.emb.table.T)

    loss = criterion(output_only_masked, y_only_masked)

    if train_mode:
        loss.backward()
        model.emb.table.grad[model.emb.padding_idx] = torch.zeros_like(model.emb.table.grad[model.emb.padding_idx])
        optim.step()

    loss_item = loss.item()

    output_labels = output_only_masked.argmax(dim=-1)
    a = torch.count_nonzero(output_labels == y_only_masked)
    acc = a.item() / y_only_masked.size(0)
    # TODO: Add macro-average-acc?

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


def save_model(model_dir, model_files, keep_last_models, epoch):
    model_file = os.path.join(model_dir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), model_file)
    model_files.append(model_file)
    if len(model_files) > keep_last_models:
        for model_file in model_files[:-keep_last_models]:
            try:
                os.remove(model_file)
            except Exception as e:
                get_logger().warning('Deleting model file fails. %s, %s', model_file, e)
        model_files = model_files[-keep_last_models:]
    
    return model_files


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', type=str, default='configs/config_ln_encoder.yaml')
    ap.add_argument('-d', '--model_dir', type=str, default='')
    ap.add_argument('-r', '--resume', default=False, action='store_true')
    args = ap.parse_args()

    config_file = args.config
    model_dir = args.model_dir
    resume = args.resume

    if not os.path.exists(config_file):
        get_logger().error('config file %s not exists.', config_file)
        sys.exit()

    with open(config_file, 'rt') as f:
        config = yaml.load(f, yaml.SafeLoader)

    keep_last_models = config['keep_last_models']
    step_save_ckpt = config['step_save_ckpt']
    cuda_index = config['cuda_index']
    epoch = config['epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    d_model = config['d_model']
    h = config['h']
    ff = config['ff']
    n_layers = config['n_layers']
    p_dropout = config['p_dropout']
    seq_len = config['seq_len']
    vocab_file = config['vocab_file']
    vocab_start_token_id = config['vocab_start_token_id']
    train_dataset_files = config['train_dataset_files']
    shuffle_dataset_on_load = config['shuffle_dataset_on_load']
    valid_dataset_files = config['valid_dataset_files']

    assert keep_last_models > 0, 'keep_last_models config value must be greater than 0.'

    if not model_dir and not resume:
        model_dir = f'output/model_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    elif resume and not model_dir:
        get_logger().error('No model_dir arg for resume.')
        sys.exit()

    with open(vocab_file, 'rt') as f:
        vocab = json.load(f)

    if os.path.exists(model_dir):
        if not resume:
            get_logger().error('model directory (%s) already exists.', model_dir)
            sys.exit()
    else:
        if resume:
            get_logger().error('model directory (%s) not exists.', model_dir)
            sys.exit()
        else:
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy(config_file, model_dir)

    init_logger(os.path.join(model_dir, 'log.log'))
    get_logger().info('Training started.')
    separator = '='*80
    txt = 'configuration information\n'
    txt += f'{separator}\n'
    for k, v in config.items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}'
    get_logger().info(txt)

    # TODO: use parallel gpu
    device = get_torch_device(cuda_index)

    get_logger().info('Used device type: %s', device.type)
    model = lm_encoder(
        d_model, h, ff, n_layers, len(vocab),
        padding_idx=vocab['__PAD__'], dropout_p=p_dropout
    )
    model.to(device=device)

    txt = f'model information\n{separator}\n'
    num_params = 0
    for name, params in model.named_parameters():
        txt += f'{name}, {params.shape}\n'
        # This has the problem of reinitializing all the initialized values of the modules. Implemented in the constructor of each module.
        # if params.dim() > 1:
        #     torch.nn.init.xavier_uniform_(params)
        if params.requires_grad is True:
            num_params += params.numel()
    txt += (
        f"{separator}\n"
        f"{model}\n"
        f"{separator}\n"
        f"Number of parameters: {num_params:,}\n"
        f"{separator}")
    
    get_logger().info(txt)
    
    # dummy_input_tensor = torch.randint(100, [seq_len, d_model], device=device)
    # try:
    #     # with torch.no_grad():
    #     model.eval()
    #     model(dummy_input_tensor)
    # except torch.cuda.OutOfMemoryError as e:
    #     get_logger().error(e)
    #     sys.exit(1)
    # del dummy_input_tensor

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if resume:
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        def get_epoch_of_model_file(x):
            model_epoch = re.search(r'(?<=^model_)\d+(?=.pt$)', os.path.basename(x))
            return 0 if not model_epoch else int(model_epoch.group())
        
        model_files = sorted(glob.glob(os.path.join(model_dir, 'model_*.pt')), key=get_epoch_of_model_file)
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
    else:
        model_files = []
        start_epoch = 0
        step = 0

    # optimized_model = torch.compile(model,)
    # del model
    # torch.cuda.empty_cache()
    # model = optimized_model

    get_logger().info('Loading dataset...')
    train_dataloader = mlm_dataloader(
        train_dataset_files, vocab, vocab_start_token_id,
        seq_len, batch_size, shuffle_dataset_on_load, dynamic_masking=True
    )
    valid_dataloader = mlm_dataloader(
        valid_dataset_files, vocab, vocab_start_token_id,
        seq_len, batch_size, dynamic_masking=False
    )
    
    datasets = ''
    if train_dataloader:
        datasets += f'trains: {len(train_dataloader.dataset)}, steps per epoch: {len(train_dataloader)} '
    if valid_dataloader:
        datasets += f'valids: {len(valid_dataloader.dataset)}, steps per epoch: {len(valid_dataloader)}'
    get_logger().info('Dataset loaded. %s', datasets)

    dataloader = train_dataloader or valid_dataloader
    sw = SummaryWriter(os.path.join(model_dir, 'logs'))
    sw.add_graph(model, dataloader.dataset[0][0].to(device))
    
    torch.cuda.empty_cache()

    sleep_between_step = .0
    train_loss, train_acc, val_loss, val_acc = .0, .0, .0, .0
    train_interval_loss, train_interval_acc = .0, .0
    logging_interval = 20
    elapsed_train = 0

    try:
        for current_epoch in range(start_epoch+1, epoch+1):
            training_sec_per_epoch = {}
            if train_dataloader is not None:
                for train_data in train_dataloader:
                    step += 1
                    model.train()
                    training_started = time.time()
                    train_loss, train_acc = run_step(train_data, model, loss_fn, optim, True, device)
                    elapsed_train += time.time() - training_started
                    train_interval_loss += train_loss
                    train_interval_acc += train_acc
                    if step % logging_interval == 0:
                        iterval_loss = train_interval_loss / logging_interval
                        interval_acc = train_interval_acc / logging_interval
                        get_logger().info('%d/%d training loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                                          current_epoch, step, round(iterval_loss, 4), round(interval_acc, 4), round(elapsed_train,2))
                        sw.add_scalar('elapsed/train', elapsed_train, step)
                        sw.add_scalar('Loss/train', iterval_loss, step)
                        sw.add_scalar('Acc/train', interval_acc, step)
                        elapsed_train = 0
                        train_interval_loss = .0
                        train_interval_acc = .0

                    if step > 1 and step % step_save_ckpt == 0:
                        torch.save(
                            {
                                'step': step,
                                'epoch': current_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optim.state_dict()
                            },
                            os.path.join(model_dir, 'checkpoint.pt')
                        )
                        model_files = save_model(model_dir, model_files, keep_last_models, step)
                        get_logger().info('checkpoint saved at %d/%d', current_epoch, step)

                        if valid_dataloader is not None:
                            get_logger().info('%d/%d Start to validation', current_epoch, step)
                            model.eval()
                            valid_started = time.time()
                            with torch.no_grad():
                                val_loss, val_acc = run_epoch(valid_dataloader, model, loss_fn, None, False, sleep_between_step, device)
                            elapsed_valid = time.time() - valid_started
                            sw.add_scalar('elapsed/valid', elapsed_valid, step)
                            sw.add_scalar('Loss/valid', val_loss, step)
                            sw.add_scalar('Acc/valid', val_acc, step)
                            get_logger().info('%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                                              current_epoch, step, round(val_loss, 4), round(val_acc, 4), round(elapsed_valid, 2))

                            if train_dataloader is None:
                                break

            get_logger().info('%s/%s training a epoch finished.', current_epoch, step)
            # sw.add_scalars('elapsed_sec_per_epoch', training_sec_per_epoch, i+1)
            # get_logger().info(f'epoch {i+1}. train_loss: {round(train_loss,4):>8}, train_acc: {round(train_acc, 4):>8}, val_loss: {round(val_loss,4):>8}, val_acc: {round(val_acc, 4):>8}, elapsed: {round(time.time()-t1,3)}s')
    except KeyboardInterrupt:
        get_logger().info("Training stopped.")
    except Exception as e:
        get_logger().error('Exception occured during training. %s', e)
    else:
        get_logger().info("All training finished.")

    sw.close()
