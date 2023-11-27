import sys
import os
import argparse
import json
import time
from functools import partial
from copy import deepcopy
from itertools import count
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ray import tune
from ray.air import session
from ray.air import RunConfig
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from hs_aiteam_pkgs.util.logger import init_logger, get_logger
from hs_aiteam_pkgs.util.signal_handler import SigTermException
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import tweet_disaster_dataset
from models.tweet_disaster_model import TweetDisasterClassifierCNN
from checkpoint import save_checkpoint, save_model
from training_iter import run_step_fine_tuning, run_epoch
import configuration
import configuration_fine_tuning

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


def train_n_val(config, dataset, vocab, pre_train_config, pre_trained_model, train_config):
    unfreeze_last_layers = config['unfreeze_last_layers']
    remove_last_layers = config['remove_last_layers']
    learning_rate = config['learning_rate']
    conv_filters = config['conv_filters']
    weight_decay = config['weight_decay']
    p_dropout = config['p_dropout']
    kernel_sizes = config['kernel_sizes']

    collate_fn = partial(tweet_disaster_dataset.collate_fn,
                         max_seq=train_config.seq_len, padding_idx=vocab['__PAD__'])
    train_dataloader = DataLoader(
        dataset['train'], train_config.batch_size, train_config.shuffle_dataset_on_load,
        num_workers=0, collate_fn=collate_fn,)
        # worker_init_fn=mlm_dataset.worker_init)
    valid_dataloader = DataLoader(
        dataset['valid'], train_config.batch_size, train_config.shuffle_dataset_on_load,
        num_workers=0, collate_fn=collate_fn,)
        # worker_init_fn=mlm_dataset.worker_init)
    _model_dir = os.path.join(session.get_trial_dir(), 'customs')
    os.makedirs(_model_dir, exist_ok=True)

    _config = train_config
    new_config = deepcopy(_config)
    new_config.learning_rate = learning_rate
    new_config.unfreeze_last_layers = unfreeze_last_layers
    new_config.remove_last_layers = remove_last_layers
    new_config.conv_filters = conv_filters
    new_config.kernel_sizes = kernel_sizes
    new_config.weight_decay = weight_decay
    new_config.p_dropout = p_dropout

    if _config != new_config:
        new_conf = os.path.join(_model_dir, 'train_config.yaml')
        with open(new_conf, 'wt', encoding='utf8') as f:
            yaml.dump(configuration_fine_tuning.convert_to_dict(new_config), f, sort_keys=False)

    if _config != new_config:
        _config = new_config

    init_logger(os.path.join(_model_dir, 'log.log'))
    get_logger().info('Training started.')
    separator = '='*80
    txt = 'configuration information\n'
    txt += f'{separator}\n'
    for k, v in configuration.convert_to_dict(pre_train_config).items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}\n'
    for k, v in configuration_fine_tuning.convert_to_dict(_config).items():
        txt += f'{k:<22}: {v}\n'
    txt += f'{separator}'
    get_logger().info(txt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    get_logger().info('Used device type: %s', device.type)
    origin_model = TweetDisasterClassifierCNN.from_pretrained(
        pre_trained_model, pre_train_config, _config.unfreeze_last_layers,
        _config.remove_last_layers, _config.conv_filters, _config.kernel_sizes,
        dropout_p=_config.p_dropout)

    if _config.compile_model:
        model = torch.compile(origin_model)
        get_logger().info('model compiled.')
    else:
        model = origin_model
    model.to(device=device)
    model.train()
    # txt = f'model information\n{separator}\n'
    num_params = 0
    for name, params in model.named_parameters():
        txt += f'{name}, {params.shape}\n'
        if params.requires_grad is True:
            num_params += params.numel()

    get_logger().info("Number of model parameters: %s", format(num_params, ','))

    # Expand GPU memory of model before load dataset.
    dummy_tensor = torch.randint(
        0, len(vocab)-1, [_config.batch_size, _config.seq_len], device=device)
    try:
        model(dummy_tensor)
    except torch.cuda.OutOfMemoryError as oom_exception:
        get_logger().error('CUDA out of memory before load dataset: %s', oom_exception)
        sys.exit(1)
    else:
        del dummy_tensor

    loss_fn = torch.nn.BCELoss()
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

    epoch_train_loss = epoch_train_acc = .0
    it = range(1, _config.epoch+1) if _config.epoch is not None else count(1)
    step = 0
    iters = len(train_dataloader)
    # max_acc = 0
    # min_loss = float('inf')
    try:
        for current_epoch in it:
            summary_writer.add_scalar('epoch', current_epoch, step+1)
            for train_data in train_dataloader:
                step += 1
                training_started = time.time()
                train_loss, train_acc = run_step_fine_tuning(
                    train_data, model, loss_fn, optim, True, device)
                elapsed_train += time.time() - training_started
                train_interval_loss += train_loss
                train_interval_acc += train_acc
                epoch_train_loss += train_loss
                epoch_train_acc += train_acc

                # change lr.
                if scheduler:
                    scheduler.step(step / iters)

                # logging per 20 steps.
                if step % logging_interval == 0:
                    iterval_loss = train_interval_loss / logging_interval
                    interval_acc = train_interval_acc / logging_interval
                    # min_loss = min(train_interval_loss, min_loss)
                    # max_acc = max(interval_acc, max_acc)
                    get_logger().info(
                        '%d/%d training loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(iterval_loss, 4),
                        round(interval_acc, 4), round(elapsed_train, 2))
                    summary_writer.add_scalar('elapsed/train', elapsed_train, step)
                    summary_writer.add_scalar('Loss/train', iterval_loss, step)
                    summary_writer.add_scalar('Acc/train', interval_acc, step)
                    summary_writer.add_scalar('learning_rate', optim.param_groups[0]["lr"], step)
                    elapsed_train = 0
                    train_interval_loss = .0
                    train_interval_acc = .0

            # save checkpoint and validate.
            if step > 1:
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
                            sleep_between_step, device, True)
                    elapsed_valid = time.time() - valid_started
                    summary_writer.add_scalar('elapsed/valid', elapsed_valid, step)
                    summary_writer.add_scalar('Loss/valid', val_loss, step)
                    summary_writer.add_scalar('Acc/valid', val_acc, step)
                    get_logger().info(
                        '%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(val_loss, 4),
                        round(val_acc, 4), round(elapsed_valid, 2))
                    results = {
                        'train_loss': epoch_train_loss/len(train_dataloader),
                        'train_acc': epoch_train_acc/len(train_dataloader),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                    session.report(results)
                    model.train()
                    if train_dataloader is None:
                        break
            epoch_train_loss = .0
            epoch_train_acc = .0

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


def main(
        pre_trained_config_file,
        pre_trained_model_file,
        fine_tuning_config_file,
        _model_dir,
        _restore_dir):
    try:
        pre_trained_config = configuration.load_config_file(pre_trained_config_file)
        configuration.validate_config(pre_trained_config)
    except TypeError as type_e:
        get_logger().error('key of configuration is missing. %s', type_e)
        sys.exit(1)
    except AssertionError as assert_e:
        get_logger().error(assert_e)
        sys.exit(1)

    with open(pre_trained_config.vocab_file, 'rt', encoding='utf8') as f:
        vocab = json.load(f)

    try:
        fine_tuning_config = configuration_fine_tuning.load_config_file(fine_tuning_config_file)
        configuration_fine_tuning.validate_config(fine_tuning_config)
    except TypeError as type_e:
        get_logger().error('key of configuration is missing. %s', type_e)
        sys.exit(1)
    except AssertionError as assert_e:
        get_logger().error(assert_e)
        sys.exit(1)

    dataset_dict = {}
    dataset_dict['train'] = tweet_disaster_dataset.TweetDisasterDataset(
        fine_tuning_config.train_dataset_file,
        fine_tuning_config.train_dataset_label_file,
        vocab, fine_tuning_config.seq_len)

    dataset_dict['valid'] = tweet_disaster_dataset.TweetDisasterDataset(
        fine_tuning_config.valid_dataset_file,
        fine_tuning_config.valid_dataset_label_file,
        vocab, fine_tuning_config.seq_len)

    config = {
        "unfreeze_last_layers": tune.grid_search([0, 1]),
        "remove_last_layers": tune.grid_search([1, 2]),
        "learning_rate": tune.grid_search([1e-4, 1e-5]),
        'conv_filters': tune.grid_search([100]),
        'kernel_sizes': tune.grid_search([[3, 4, 5]]),
        'weight_decay': tune.grid_search([0, 0.1, 0.2]),
        'p_dropout': tune.grid_search([0.2, 0.5]),
    }
    train_func = tune.with_resources(
            tune.with_parameters(train_n_val,
                                 dataset=dataset_dict,
                                 vocab=vocab,
                                 pre_train_config=pre_trained_config,
                                 pre_trained_model=pre_trained_model_file,
                                 train_config=fine_tuning_config),
            resources={'gpu': 1}
        )
    if _restore_dir:
        if not tune.Tuner.can_restore(_restore_dir):
            get_logger().error('can\'t restore. %s', _restore_dir)
            sys.exit(1)
        tuner = tune.Tuner.restore(_restore_dir, trainable=train_func)
    else:
        # scheduler = ASHAScheduler()
        # algo = HyperOptSearch(metric='acc', mode='max')
        # stopper = TrialPlateauStopper('val_acc', std=0.005, num_results=5, grace_period=5)
        tuner = tune.Tuner(
            train_func,
            tune_config=tune.TuneConfig(
                metric="val_acc",
                mode="max",
                # scheduler=scheduler,
                # search_alg=algo,
                num_samples=1,
                max_concurrent_trials=1
            ),
            run_config=RunConfig(storage_path=_model_dir if _model_dir else None),
                                #  stop=stopper),
            param_space=config,
        )
    results = tuner.fit()

    best_result = results.get_best_result("val_acc", "max")

    print(f"Best trial config: {best_result.config}")
    # print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final train accuracy: {best_result.metrics['train_acc']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['val_acc']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--pre_trained_config', type=str,
                    default='configs/config_ln_encoder.yaml')
    ap.add_argument('-p', '--pre_trained_model', type=str, required=True)
    ap.add_argument('-f', '--fine_tuning_config', type=str,
                    default='config/config_fine_tuning.yaml')
    ap.add_argument('-d', '--model_dir', type=str, default='')
    ap.add_argument('-r', '--restore_dir', type=str, default='')
    args = ap.parse_args()

    _pre_trained_config = args.pre_trained_config
    _pre_trained_model = args.pre_trained_model
    _fine_tuning_config = args.fine_tuning_config
    _model_dir = args.model_dir
    _restore_dir = args.restore_dir

    if not os.path.exists(_pre_trained_config):
        get_logger().error('config file %s not exists.', _pre_trained_config)
        sys.exit()
    if not os.path.exists(_pre_trained_model):
        get_logger().error('config file %s not exists.', _pre_trained_model)
        sys.exit()
    if not os.path.exists(_fine_tuning_config):
        get_logger().error('config file %s not exists.', _fine_tuning_config)
        sys.exit()
    main(
        _pre_trained_config,
        _pre_trained_model,
        _fine_tuning_config,
        _model_dir,
        _restore_dir)
