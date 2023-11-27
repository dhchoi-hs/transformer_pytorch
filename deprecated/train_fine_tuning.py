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
from torch.utils.data import DataLoader, Subset
from hs_aiteam_pkgs.util.logger import init_logger, get_logger
from hs_aiteam_pkgs.util.signal_handler import SigTermException, catch_kill_signal
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import tweet_disaster_dataset
from model.utils.get_torch_device import get_torch_device
import configuration
import configuration_fine_tuning
from checkpoint import load_ckpt, save_checkpoint, save_model
from training_iter import run_step_fine_tuning, run_epoch
from models.tweet_disaster_model import TweetDisasterClassifierCNN

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
catch_kill_signal()


def main(_pre_trained_config, pre_trained_model_file, _fine_tuning_config, _model_dir, _resume):
    try:
        pre_train_config = configuration.load_config_file(_pre_trained_config)
        configuration.validate_config(pre_train_config)
    except TypeError as type_e:
        get_logger().error('key of pre trained configuration is missing. %s', type_e)
        sys.exit(1)
    except AssertionError as assert_e:
        get_logger().error('error loading pre train config.: %s', assert_e)
        sys.exit(1)

    try:
        fine_tuning_config = configuration_fine_tuning.load_config_file(_fine_tuning_config)
        configuration_fine_tuning.validate_config(fine_tuning_config)
    except TypeError as type_e:
        get_logger().error('key of pre fine tuning configuration is missing. %s', type_e)
        sys.exit(1)
    except AssertionError as assert_e:
        get_logger().error('error loading fine tuning config.: %s', assert_e)
        sys.exit(1)

    if not _model_dir and not _resume:
        _model_dir = f'output/fine_tuning_model_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    elif _resume and not _model_dir:
        get_logger().error('No model_dir arg for resume.')
        sys.exit()

    with open(pre_train_config.vocab_file, 'rt', encoding='utf8') as f:
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
            shutil.copy(_pre_trained_config, _model_dir)
            shutil.copy(_fine_tuning_config, _model_dir)

    init_logger(os.path.join(_model_dir, 'log.log'))
    get_logger().info('Training started.')
    separator = '='*80
    txt = 'configuration information\n'
    txt += f'{separator}\n'
    for k, v in configuration.convert_to_dict(pre_train_config).items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}\n'
    for k, v in configuration_fine_tuning.convert_to_dict(fine_tuning_config).items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}'
    get_logger().info(txt)

    device = get_torch_device(fine_tuning_config.cuda_index)
    get_logger().info('Used device type: %s', device.type)

    origin_model = TweetDisasterClassifierCNN.from_pretrained(
        pre_trained_model_file, pre_train_config, fine_tuning_config.unfreeze_last_layers,
        fine_tuning_config.remove_last_layers, fine_tuning_config.conv_filters,
        kernel_sizes=fine_tuning_config.kernel_sizes, dropout_p=fine_tuning_config.p_dropout)

    if fine_tuning_config.compile_model:
        model = torch.compile(origin_model)
        get_logger().info('model compiled.')
    else:
        model = origin_model
    model.to(device=device)
    model.train()

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

    loss_fn = torch.nn.BCELoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=fine_tuning_config.learning_rate,
                             weight_decay=fine_tuning_config.weight_decay)
    scheduler = create_lr_scheduler(optim, fine_tuning_config.lr_scheduler,
                                    **fine_tuning_config.lr_scheduler_kwargs)

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
        0, len(vocab)-1, [fine_tuning_config.batch_size, fine_tuning_config.seq_len], device=device)
    try:
        model(dummy_tensor)
    except torch.cuda.OutOfMemoryError as oom_exception:
        get_logger().error('CUDA out of memory before load dataset: %s', oom_exception)
        sys.exit(1)
    else:
        del dummy_tensor

    get_logger().info('Loading dataset...')
    collate_fn = partial(tweet_disaster_dataset.collate_fn,
                         max_seq=fine_tuning_config.seq_len, padding_idx=vocab['__PAD__'])
    dataset = tweet_disaster_dataset.TweetDisasterDataset(
        fine_tuning_config.train_dataset_file,
        fine_tuning_config.train_dataset_label_file,
        vocab, fine_tuning_config.seq_len)
    dataloader = DataLoader(
        dataset, fine_tuning_config.batch_size,
        fine_tuning_config.shuffle_dataset_on_load,
        collate_fn=collate_fn,)

    dataset_text = ''
    if dataloader:
        iters = len(dataloader)
        dataset_text += f'trains: {len(dataloader.dataset)}, steps per epoch: {iters} '
    get_logger().info('Dataset loaded. %s', dataset_text)
    del dataloader

    o = tweet_disaster_dataset.KFold(5, dataset)

    for train_indices, valid_indices in o:
        train_dataloader = DataLoader(
            Subset(dataset, train_indices), fine_tuning_config.batch_size,
            fine_tuning_config.shuffle_dataset_on_load, collate_fn=collate_fn)
        valid_dataloader = DataLoader(
            Subset(dataset, valid_indices), fine_tuning_config.batch_size,
            fine_tuning_config.shuffle_dataset_on_load, collate_fn=collate_fn)
        for t in train_dataloader:
            print(t)
        for t in valid_dataloader:
            print(t)

    purge_step = None if not _resume else step
    sw = SummaryWriter(_model_dir, purge_step=purge_step)

    train_loss = train_acc = val_loss = val_acc = .0
    train_interval_loss = train_interval_acc = .0
    logging_interval = 20
    elapsed_train = 0

    try:
        it = count(start_epoch+1) if fine_tuning_config.epoch is None else \
            range(start_epoch+1, fine_tuning_config.epoch+1)
        for current_epoch in it:
            sw.add_scalar('epoch', current_epoch, step+1)
            for train_data in train_dataloader:
                step += 1
                training_started = time.time()
                train_loss, train_acc = run_step_fine_tuning(
                    train_data, model, loss_fn, optim, True, device)
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
            if step > 1:
                save_checkpoint(origin_model, os.path.join(_model_dir, 'checkpoint.pt'),
                                step, current_epoch, optim, scheduler)
                model_files = save_model(origin_model, _model_dir, model_files,
                                            fine_tuning_config.keep_last_models, step)
                get_logger().info('checkpoint saved at %d/%d', current_epoch, step)

                if valid_dataloader is not None:
                    get_logger().info('%d/%d Start to validation', current_epoch, step)
                    model.eval()
                    valid_started = time.time()
                    with torch.no_grad():
                        val_loss, val_acc = run_epoch(
                            valid_dataloader, model, loss_fn, None, False,
                            device, True)
                    elapsed_valid = time.time() - valid_started
                    sw.add_scalar('elapsed/valid', elapsed_valid, step)
                    sw.add_scalar('Loss/valid', val_loss, step)
                    sw.add_scalar('Acc/valid', val_acc, step)
                    get_logger().info(
                        '%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(val_loss, 4), round(val_acc, 4),
                        round(elapsed_valid, 2))
                    model.train()

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
    ap.add_argument('-c', '--pre_trained_config', type=str, default='configs/config_ln_encoder.yaml')
    ap.add_argument('-p', '--pre_trained_model', type=str, required=True)
    ap.add_argument('-f', '--fine_tuning_config', type=str, default='config/config_fine_tuning.yaml')
    ap.add_argument('-d', '--model_dir', type=str, default='')
    ap.add_argument('-r', '--resume', default=False, action='store_true')
    args = ap.parse_args()

    pre_trained_config = args.pre_trained_config
    pre_trained_model = args.pre_trained_model
    fine_tuning_config = args.fine_tuning_config
    model_dir = args.model_dir
    resume = args.resume

    if not os.path.exists(pre_trained_config):
        get_logger().error('pre trained config file %s not exists.', pre_trained_config)
        sys.exit()

    if not os.path.exists(pre_trained_model):
        get_logger().error('pre trained model file %s not exists.', pre_trained_model)
        sys.exit()

    if not os.path.exists(fine_tuning_config):
        get_logger().error('fine tuning config file %s not exists.', fine_tuning_config)
        sys.exit()
    main(pre_trained_config, pre_trained_model, fine_tuning_config, model_dir, resume)
