import sys
import os
import argparse
import json
import time
from copy import deepcopy
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import ray
from ray import tune
from ray.air import session
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
from hs_aiteam_pkgs.util.logger import init_logger, get_logger
from hs_aiteam_pkgs.util.signal_handler import SigTermException
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import mlm_dataset
from models.lm_encoder import lm_encoder
from checkpoint import save_checkpoint, save_model
from training_iter import run_step, run_epoch
import configuration


torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


def train_n_val(config, train_dataset, valid_dataset, vocab, train_config, epochs):
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    layer = config['layer']
    ff = config['ff']
    h = config['h']
    d_model = config['d_model']

    collate_fn = mlm_dataset.create_collate_fn(train_config.seq_len, vocab['__PAD__'])

    train_dataloader = DataLoader(
        train_dataset, batch_size, train_config.shuffle_dataset_on_load,
        num_workers=0, collate_fn=collate_fn,)
        # worker_init_fn=mlm_dataset.worker_init)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size, train_config.shuffle_dataset_on_load,
        num_workers=0, collate_fn=collate_fn,)
        # worker_init_fn=mlm_dataset.worker_init)
    _model_dir = os.path.join(session.get_trial_dir(), 'customs')
    os.makedirs(_model_dir, exist_ok=True)

    _config = train_config
    new_config = deepcopy(_config)
    new_config.epoch = epochs
    new_config.learning_rate = learning_rate
    new_config.batch_size = batch_size
    new_config.n_layers = layer
    new_config.ff = ff
    new_config.h = h
    new_config.d_model = d_model

    if _config != new_config:
        new_conf = os.path.join(_model_dir, 'train_config.yaml')
        with open(new_conf, 'wt', encoding='utf8') as f:
            yaml.dump(configuration.convert_to_dict(new_config), f, sort_keys=False)

    if _config != new_config:
        _config = new_config

    init_logger(os.path.join(_model_dir, 'log.log'))
    get_logger().info('Training started.')
    separator = '='*80
    txt = 'configuration information\n'
    txt += f'{separator}\n'
    for k, v in configuration.convert_to_dict(_config).items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}'
    get_logger().info(txt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    get_logger().info('Used device type: %s', device.type)
    origin_model = lm_encoder(
        _config.d_model, _config.h, _config.ff, _config.n_layers, len(vocab),
        padding_idx=vocab['__PAD__'], dropout_p=_config.p_dropout,
        activation=_config.activation
    )

    if _config.compile_model:
        model = torch.compile(origin_model)
        get_logger().info('model compiled.')
    else:
        model = origin_model
    model.to(device=device)

    # txt = f'model information\n{separator}\n'
    num_params = 0
    for name, params in model.named_parameters():
        txt += f'{name}, {params.shape}\n'
        if params.requires_grad is True:
            num_params += params.numel()
    # txt += (
    #     f"{separator}\n"
    #     f"{model}\n"
    #     f"{separator}\n"
    #     f"Number of parameters: {num_params:,}\n"
    #     f"{separator}")

    get_logger().info("Number of model parameters: %s", format(num_params, ','))

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=_config.learning_rate,
                             weight_decay=_config.weight_decay)
    scheduler = create_lr_scheduler(optim, _config.lr_scheduler, **_config.lr_scheduler_kwargs)

    model_files = []

    summary_writer = SummaryWriter(os.path.dirname(_model_dir))

    sleep_between_step = .0
    train_loss = train_acc = val_loss = val_acc = .0
    train_interval_loss = train_interval_acc = .0
    logging_interval = 20
    elapsed_train = 0

    last_lr = optim.param_groups[0]["lr"]
    last_written_lr = None

    step = 0
    iters = len(train_dataloader)
    max_acc = 0
    min_loss = float('inf')
    try:
        if step == 0:
            summary_writer.add_scalar('learning_rate', last_lr, step+1)
        for current_epoch in range(1, epochs+1):
            summary_writer.add_scalar('epoch', current_epoch, step+1)
            for train_data in train_dataloader:
                step += 1
                last_lr = optim.param_groups[0]["lr"]
                model.train()
                training_started = time.time()
                train_loss, train_acc = run_step(train_data, model, loss_fn,
                                                 optim, True, device)
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
                    min_loss = min(train_interval_loss, min_loss)
                    max_acc = max(interval_acc, max_acc)
                    get_logger().info(
                        '%d/%d training loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(iterval_loss, 4),
                        round(interval_acc, 4), round(elapsed_train, 2))
                    summary_writer.add_scalar('elapsed/train', elapsed_train, step)
                    summary_writer.add_scalar('Loss/train', iterval_loss, step)
                    summary_writer.add_scalar('Acc/train', interval_acc, step)
                    results = {
                        'loss': iterval_loss,
                        'acc': interval_acc
                    }
                    session.report(results)
                    if last_written_lr != last_lr:
                        summary_writer.add_scalar('learning_rate', last_lr, step)
                        last_written_lr = last_lr
                    elapsed_train = 0
                    train_interval_loss = .0
                    train_interval_acc = .0

                # save checkpoint and validate.
                if step > 1 and step % _config.step_save_ckpt == 0:
                    # save_checkpoint(
                    #     origin_model, os.path.join(_model_dir, 'checkpoint.pt'), step,
                    #     current_epoch, optim, scheduler)
                    # model_files = save_model(
                    #     origin_model, _model_dir, model_files, _config.keep_last_models, step)
                    # get_logger().info('checkpoint saved at %d/%d', current_epoch, step)

                    if valid_dataloader is not None:
                        get_logger().info('%d/%d Start to validation', current_epoch, step)
                        model.eval()
                        valid_started = time.time()
                        with torch.no_grad():
                            val_loss, val_acc = run_epoch(
                                valid_dataloader, model, loss_fn, None, False,
                                sleep_between_step, device)
                        elapsed_valid = time.time() - valid_started
                        summary_writer.add_scalar('elapsed/valid', elapsed_valid, step)
                        summary_writer.add_scalar('Loss/valid', val_loss, step)
                        summary_writer.add_scalar('Acc/valid', val_acc, step)
                        get_logger().info(
                            '%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                            current_epoch, step, round(val_loss, 4),
                            round(val_acc, 4), round(elapsed_valid, 2))

                        if train_dataloader is None:
                            break

            summary_writer.add_scalar('epoch', current_epoch, step)
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
    # finally:
    #     save_checkpoint(origin_model, os.path.join(_model_dir, 'checkpoint.pt'), step,
    #                     current_epoch, optim, scheduler)


def main(_config_file, _model_dir):
    try:
        train_config = configuration.load_config_file(_config_file)
        configuration.validate_config(train_config)
    except TypeError as type_e:
        get_logger().error('key of configuration is missing. %s', type_e)
        sys.exit(1)
    except FileNotFoundError:
        get_logger().error('Config file not exists. %s', _config_file)
        sys.exit(1)
    except AssertionError as assert_e:
        get_logger().error(assert_e)
        sys.exit(1)

    with open(train_config.vocab_file, 'rt', encoding='utf8') as f:
        vocab = json.load(f)

    train_dataset = mlm_dataset.MLMdatasetDynamic(
        train_config.train_dataset_files, vocab,
        train_config.vocab_start_token_id, train_config.seq_len,
        train_config.shuffle_dataset_on_load, train_config.train_sampling_ratio)
    valid_dataset = mlm_dataset.MLMDatasetFixed(
        train_config.valid_dataset_files, vocab,
        train_config.vocab_start_token_id, train_config.seq_len,
        train_config.shuffle_dataset_on_load, train_config.valid_sampling_ratio)

    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([128]),
        'layer': tune.choice([6, 9, 12]),
        'ff': tune.choice([2048, 3072]),
        'h': tune.choice([8, 16]),
        'd_model': tune.choice([1024, 1536])
    }
    scheduler = ASHAScheduler()

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_n_val,
                                 train_dataset=train_dataset,
                                 valid_dataset=valid_dataset, 
                                 vocab=vocab,
                                 train_config=train_config,
                                 epochs=5,),
            resources={'cpu': 2}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=100,
            max_concurrent_trials=2
        ),
        run_config=RunConfig(storage_path=_model_dir if _model_dir else None),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['acc']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', type=str, default='configs/config_ln_encoder.yaml')
    ap.add_argument('-d', '--model_dir', type=str, default='')
    args = ap.parse_args()

    config_file = args.config
    model_dir = args.model_dir

    if not os.path.exists(config_file):
        get_logger().error('config file %s not exists.', config_file)
        sys.exit()

    main(config_file, model_dir)
