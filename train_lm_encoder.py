import sys
import os
import glob
import re
import argparse
import shutil
import json
import time
from functools import partial
from itertools import count
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from hs_aiteam_pkgs.util.logger import init_logger, get_logger
from hs_aiteam_pkgs.util.signal_handler import SigTermException, catch_kill_signal
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import mlm_dataset
from models.lm_encoder import lm_encoder
from model.utils.get_torch_device import get_torch_device
import configuration
from checkpoint import load_ckpt, save_checkpoint, save_model
from training_iter import run_step, run_epoch


torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
catch_kill_signal()


def main(_config_file, _model_dir, _resume, memo):
    try:
        config = configuration.load_config_file(_config_file)
        configuration.validate_config(config)
    except TypeError as type_e:
        get_logger().error('key of configuration is missing. %s', type_e)
        sys.exit(1)
    except FileNotFoundError:
        get_logger().error('Config file not exists. %s', _config_file)
        sys.exit(1)
    except AssertionError as assert_e:
        get_logger().error(assert_e)
        sys.exit(1)

    if not _model_dir and not _resume:
        _model_dir = f'output/model_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    elif _resume and not _model_dir:
        get_logger().error('No model_dir arg for resume.')
        sys.exit()

    with open(config.vocab_file, 'rt') as f:
        vocab = json.load(f)

    if os.path.exists(_model_dir):
        if not _resume:
            get_logger().error('model directory (%s) already exists.', _model_dir)
            sys.exit()
    else:
        if _resume:
            get_logger().error('model directory (%s) not exists.', _model_dir)
            sys.exit()
        else:
            os.makedirs(_model_dir, exist_ok=True)
            shutil.copy(_config_file, _model_dir)

    init_logger(os.path.join(_model_dir, 'log.log'))
    get_logger().info('Training started.')
    separator = '='*80
    txt = 'configuration information\n'
    txt += f'{separator}\n'
    for k, v in configuration.convert_to_dict(config).items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}'
    if memo:
        txt += f'\n{"memo":<22}: {memo}\n'
        txt += f'{separator}\n'
    get_logger().info(txt)

    # TODO: use parallel gpu
    device = get_torch_device(config.cuda_index)

    get_logger().info('Used device type: %s', device.type)
    # model = lm_encoder_torch(
    #     config.d_model, config.h, config.ff, config.n_layers, len(vocab),
    #     padding_idx=vocab['__PAD__'], dropout_p=config.p_dropout,
    #     activation='gelu'
    # )
    origin_model = lm_encoder(
        config.d_model, config.h, config.ff, config.n_layers, len(vocab),
        padding_idx=vocab['__PAD__'], dropout_p=config.p_dropout,
        activation=config.activation
    )
    if config.compile_model:
        model = torch.compile(origin_model)
        get_logger().info('model compiled.')
    else:
        model = origin_model
    model.to(device=device)

    txt = f'model information\n{separator}\n'
    num_params = 0
    for name, params in model.named_parameters():
        txt += f'{name}, {params.shape}\n'
        if params.requires_grad is True:
            num_params += params.numel()
    txt += (
        f"{separator}\n"
        f"{model}\n"
        f"{separator}\n"
        f"Number of parameters: {num_params:,}\n"
        f"{separator}")

    get_logger().info(txt)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                             weight_decay=config.weight_decay)
    scheduler = create_lr_scheduler(optim, config.lr_scheduler,
                                    **config.lr_scheduler_kwargs)
    
    if _resume:
        checkpoint = load_ckpt(os.path.join(_model_dir, 'checkpoint.pt'))
        origin_model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        def get_epoch_of_model_file(x):
            model_epoch = re.search(r'(?<=^model_)\d+(?=.pt$)', os.path.basename(x))
            return 0 if not model_epoch else int(model_epoch.group())
        
        model_files = sorted(glob.glob(os.path.join(_model_dir, 'model_*.pt')),
                             key=get_epoch_of_model_file)
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        checkpoint = {}
        model_files = []
        start_epoch = 0
        step = 0

    del checkpoint

    # Expand GPU memory of model before load dataset.
    dummy_tensor = torch.randint(
        0, len(vocab)-1, [config.batch_size, config.seq_len], device=device)
    try:
        model(dummy_tensor)
    except torch.cuda.OutOfMemoryError as oom_exception:
        get_logger().error('CUDA out of memory before load dataset: %s', oom_exception)
        sys.exit(1)
    else:
        del dummy_tensor

    get_logger().info('Loading dataset...')
    collate_fn = partial(mlm_dataset.collate_fn, max_seq=config.seq_len, padding_idx=vocab['__PAD__'])
    dataset = mlm_dataset.MLMdatasetDynamic(
        config.train_dataset_files, vocab, config.vocab_start_token_id,
        config.seq_len, config.shuffle_dataset_on_load, config.train_sampling_ratio)
    train_dataloader = DataLoader(
        dataset, config.batch_size, config.shuffle_dataset_on_load, num_workers=2,
        collate_fn=collate_fn,
        worker_init_fn=mlm_dataset.worker_init)

    dataset = mlm_dataset.MLMDatasetFixed(
        config.valid_dataset_files, vocab, config.vocab_start_token_id,
        config.seq_len, config.shuffle_dataset_on_load, config.valid_sampling_ratio)
    valid_dataloader = DataLoader(
        dataset, config.batch_size, config.shuffle_dataset_on_load, num_workers=2,
        collate_fn=collate_fn,
        worker_init_fn=mlm_dataset.worker_init)

    datasets = ''
    if train_dataloader:
        iters = len(train_dataloader)
        datasets += f'trains: {len(train_dataloader.dataset)}, steps per epoch: {iters} '
    if valid_dataloader:
        datasets += f'valids: {len(valid_dataloader.dataset)}, '\
            f'steps per epoch: {len(valid_dataloader)}'
    get_logger().info('Dataset loaded. %s', datasets)

    purge_step = None if not resume else step
    sw = SummaryWriter(_model_dir, purge_step=purge_step)

    sleep_between_step = .0
    train_loss = train_acc = val_loss = val_acc = .0
    train_interval_loss = train_interval_acc = .0
    logging_interval = 20
    elapsed_train = 0

    try:
        if step == 0:
            sw.add_scalar('learning_rate', optim.param_groups[0]["lr"], step+1)
        it = count(start_epoch+1) if config.epoch is None else range(start_epoch+1, config.epoch+1)
        for current_epoch in it:
            sw.add_scalar('epoch', current_epoch, step+1)
            for train_data in train_dataloader:
                step += 1
                model.train()
                training_started = time.time()
                train_loss, train_acc = run_step(train_data, model, loss_fn, optim, True, device)
                elapsed_train += time.time() - training_started
                train_interval_loss += train_loss
                train_interval_acc += train_acc

                # change lr.
                if scheduler:
                    scheduler.step(step / iters)

                # logging per 20 steps.
                if step % logging_interval == 0:
                    iterval_loss = train_interval_loss / logging_interval
                    interval_acc = train_interval_acc / logging_interval
                    get_logger().info(
                        '%d/%d training loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(iterval_loss, 4),
                        round(interval_acc, 4), round(elapsed_train,2))
                    sw.add_scalar('elapsed/train', elapsed_train, step)
                    sw.add_scalar('Loss/train', iterval_loss, step)
                    sw.add_scalar('Acc/train', interval_acc, step)
                    sw.add_scalar('learning_rate', optim.param_groups[0]["lr"], step)
                    elapsed_train = 0
                    train_interval_loss = .0
                    train_interval_acc = .0

                # save checkpoint and validate.
                if step > 1 and step % config.step_save_ckpt == 0:
                    save_checkpoint(origin_model, os.path.join(_model_dir, 'checkpoint.pt'),
                                    step, current_epoch, optim, scheduler)
                    model_files = save_model(origin_model, _model_dir, model_files,
                                             config.keep_last_models, step)
                    get_logger().info('checkpoint saved at %d/%d', current_epoch, step)

                    if valid_dataloader is not None:
                        get_logger().info('%d/%d Start to validation', current_epoch, step)
                        model.eval()
                        valid_started = time.time()
                        with torch.no_grad():
                            val_loss, val_acc = run_epoch(
                                valid_dataloader, model,loss_fn, None, False,
                                sleep_between_step, device)
                        elapsed_valid = time.time() - valid_started
                        sw.add_scalar('elapsed/valid', elapsed_valid, step)
                        sw.add_scalar('Loss/valid', val_loss, step)
                        sw.add_scalar('Acc/valid', val_acc, step)
                        get_logger().info(
                            '%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                            current_epoch, step, round(val_loss, 4), round(val_acc, 4),
                            round(elapsed_valid, 2))

                        if train_dataloader is None:
                            break

            sw.add_scalar('epoch', current_epoch, step)
            get_logger().info('%s/%s Training a epoch finished.', current_epoch, step)
    except KeyboardInterrupt:
        get_logger().info('Training stopped by Ctrl+C.')
    except SigTermException:
        get_logger().info('Training stopped by sigterm.')
    except torch.cuda.OutOfMemoryError as oom_exception:
        get_logger().error('CUDA out of memory. :%s', oom_exception)
    except Exception as exception:
        get_logger().error('Exception occured during training. %s', exception)
    else:
        get_logger().info("All training finished.")
    finally:
        save_checkpoint(origin_model, os.path.join(_model_dir, 'checkpoint.pt'),
                        step, current_epoch, optim, scheduler)

    sw.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', type=str, default='configs/config_ln_encoder.yaml')
    ap.add_argument('-d', '--model_dir', type=str, default='')
    ap.add_argument('-r', '--resume', default=False, action='store_true')
    ap.add_argument('-m', '--memo', type=str, default='')
    args = ap.parse_args()

    config_file = args.config
    model_dir = args.model_dir
    resume = args.resume
    memo = args.memo

    if not os.path.exists(config_file):
        get_logger().error('config file %s not exists.', config_file)
        sys.exit()

    main(config_file, model_dir, resume, memo)
