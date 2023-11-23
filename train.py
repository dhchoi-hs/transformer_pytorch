import sys
import os
import argparse
from datetime import datetime
import shutil
import torch
from hs_aiteam_pkgs.util.logger import init_logger, get_logger
from hs_aiteam_pkgs.util.signal_handler import catch_kill_signal
from trainer.pre_train import PreTrainTrainer
from trainer.fine_tuning import FineTuningTrainer
from trainer.fine_tuning_cross_validation import FineTuningCrossValidationTrainer

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
catch_kill_signal()
# torch.autograd.set_detect_anomaly(True)


def _init_model_dir(model_dir, train_type, resume, config_path):
    if not model_dir and not resume:
        _model_dir = f'output/model_{train_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    elif resume and not model_dir:
        raise ValueError('No model_dir arg for resume.')
    else:
        _model_dir = model_dir

    if os.path.exists(_model_dir):
        if not resume:
            raise FileExistsError(f'model directory ({_model_dir}) already exists.')
    else:
        if resume:
            raise FileNotFoundError(f'model directory {_model_dir} not exists.')
        os.makedirs(_model_dir, exist_ok=True)
        shutil.copy(config_path, _model_dir)

    return _model_dir

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--pre_trained_config', type=str)
    ap.add_argument('-p', '--pre_trained_model', type=str)
    ap.add_argument('-f', '--fine_tuning_config', type=str)
    ap.add_argument('-d', '--model_dir', type=str, default='')
    ap.add_argument('-r', '--resume', default=False, action='store_true')
    ap.add_argument('-m', '--memo', type=str, default='')
    ap.add_argument('-t', '--train_type', type=str,
                    choices=['pre-train', 'fine-tuning', 'fine-tuning-cv'],
                    default='pre-train')

    args = ap.parse_args()

    pre_train_config_file = args.pre_trained_config
    pre_trained_model = args.pre_trained_model
    fine_tuning_config = args.fine_tuning_config
    model_dir = args.model_dir
    resume = args.resume
    memo = args.memo

    if args.train_type == 'pre-train':
        _model_dir = _init_model_dir(model_dir, args.train_type, resume, pre_train_config_file)
    elif args.train_type in ('fine-tuning', 'fine-tuning-cv'):
        _model_dir = _init_model_dir(model_dir, args.train_type, resume, fine_tuning_config)
    else:
        _model_dir = _init_model_dir(model_dir, args.train_type, resume, pre_train_config_file)

    init_logger(os.path.join(_model_dir, 'log.log' if not resume else 'log_resume.log'))

    if not os.path.exists(pre_train_config_file):
        get_logger().error('config file %s not exists.', pre_train_config_file)
        sys.exit()

    if args.train_type == 'pre-train':
        trainer = PreTrainTrainer(pre_train_config_file, _model_dir, resume, memo)
    elif args.train_type == 'fine-tuning':
        trainer = FineTuningTrainer(
            fine_tuning_config,
            pre_trained_model,
            pre_train_config_file,
            _model_dir,
            resume,
            memo)
    elif args.train_type == 'fine-tuning-cv':
        trainer = FineTuningCrossValidationTrainer(
            fine_tuning_config,
            pre_trained_model,
            pre_train_config_file,
            _model_dir,
            resume,
            memo)
    else:
        get_logger().error('Invalid train type. %s', args.train_type)
        sys.exit()

    trainer.initialize_train()
    trainer.fit()
