import os
import glob
import re
import json
import time
import shutil
from functools import partial
from itertools import count
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from hs_aiteam_pkgs.util.logger import get_logger
from hs_aiteam_pkgs.util.signal_handler import SigTermException
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import mlm_dataset
from models.lm_encoder import lm_encoder
from model.utils.get_torch_device import get_torch_device
import configuration
from checkpoint import load_ckpt, save_checkpoint, save_model
from training_iter import run_step, run_epoch


class PreTrainTrainer:
    def __init__(self, _config_file, _model_dir, _resume, memo) -> None:
        self.config_path = _config_file
        self.model_dir = _model_dir
        self.resume = _resume
        self.memo = memo

        self.config = None
        self.vocab = None
        self.device = None

    def initialize_train(self):
        self.config = self._init_config()
        self.vocab = self._load_vocab()
        if self.config.cuda_index is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.cuda_index)
            self.device = get_torch_device(0)
        else:
            self.device = get_torch_device(None)

    def _init_config(self):
        config = configuration.load_config_file(self.config_path)
        configuration.validate_config(config)
        return config

    def _log_config(self):
        separator = '='*80
        txt = 'configuration information\n'
        txt += f'{separator}\n'
        for k, v in configuration.convert_to_dict(self.config).items():
            txt += f'{k:<22}: {v}\n'
        txt += f'{separator}'
        if self.memo:
            txt += f'\n{"memo":<22}: {self.memo}\n'
            txt += f'{separator}\n'
        get_logger().info(txt)

    def _load_model(self):
        model = lm_encoder(
            self.config.d_model, self.config.h, self.config.ff, self.config.n_layers,
            len(self.vocab), padding_idx=self.vocab['__PAD__'], dropout_p=self.config.p_dropout,
            activation=self.config.activation
        )

        return model

    @staticmethod
    def _get_epoch_of_model_file(x):
        model_epoch = re.search(r'(?<=^model_)\d+(?=.pt$)', os.path.basename(x))
        return 0 if not model_epoch else int(model_epoch.group())

    def _load_datasets(self):
        collate_fn = partial(mlm_dataset.collate_fn, max_seq=self.config.seq_len, padding_idx=self.vocab['__PAD__'])
        dataset = mlm_dataset.MLMdatasetDynamic(
            self.config.train_dataset_files, self.vocab, self.config.vocab_start_token_id,
            self.config.seq_len, self.config.shuffle_dataset_on_load, self.config.train_sampling_ratio)
        train_dataloader = DataLoader(
            dataset, self.config.batch_size, self.config.shuffle_dataset_on_load, num_workers=2,
            collate_fn=collate_fn,
            worker_init_fn=mlm_dataset.worker_init)

        dataset = mlm_dataset.MLMDatasetFixed(
            self.config.valid_dataset_files, self.vocab, self.config.vocab_start_token_id,
            self.config.seq_len, self.config.shuffle_dataset_on_load, self.config.valid_sampling_ratio)
        valid_dataloader = DataLoader(
            dataset, self.config.batch_size, self.config.shuffle_dataset_on_load, num_workers=2,
            collate_fn=collate_fn,
            worker_init_fn=mlm_dataset.worker_init)
        
        return train_dataloader, valid_dataloader

    def _init_loss_fn(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn.to(device=self.device)
        return loss_fn

    def _init_optimizer(self, model):
        optim = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        return optim

    def _train_step(self, _data, model, criterion, optim=None):
        return run_step(_data, model, criterion, optim, True, self.device)

    def _valid_epoch(self, dataset, model, criterion):
        return run_epoch(dataset, model, criterion, None, False, self.device, False)

    def _train_loop(self, model, origin_model, train_dataloader, valid_dataloader):
        '''
        model: model for training. It may or may not be compiled.
        origin_model: not compiled model. It's used to save and load state dict.
        '''
        loss_fn = self._init_loss_fn()
        optim = self._init_optimizer(model)
        scheduler = create_lr_scheduler(optim, self.config.lr_scheduler,
                                        **self.config.lr_scheduler_kwargs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optim,
        #     mode='min',
        #     factor=0.2,
        #     patience=3,
        #     threshold=1e-2
        # )

        if self.resume:
            checkpoint = load_ckpt(os.path.join(self.model_dir, 'checkpoint.pt'))
            origin_model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])

            model_files = sorted(glob.glob(os.path.join(self.model_dir, 'model_*.pt')),
                                 key=self._get_epoch_of_model_file)
            start_epoch = checkpoint['epoch']
            step = checkpoint['step']
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            del checkpoint
        else:
            model_files = []
            start_epoch = 0
            step = 0

        tb_writer = SummaryWriter(
            self.model_dir, purge_step=None if not self.resume else step)

        iters = len(train_dataloader)

        train_interval_loss = train_interval_acc = .0
        train_epoch_loss = train_epoch_acc = .0
        logging_interval = 20
        elapsed_train = 0

        if step == 0:
            tb_writer.add_scalar('learning_rate', optim.param_groups[0]["lr"], step+1)
        try:
            for current_epoch in count(start_epoch+1) \
                    if self.config.epoch is None else range(start_epoch+1, self.config.epoch+1):
                tb_writer.add_scalar('epoch', current_epoch, step+1)
                train_epoch_loss = train_epoch_acc = .0
                for train_data in train_dataloader:
                    step += 1
                    model.train()
                    training_started = time.time()
                    metrics = self._train_step(train_data, model, loss_fn, optim)
                    elapsed_train += time.time() - training_started
                    train_interval_loss += metrics['loss']
                    train_interval_acc += metrics['acc']
                    train_epoch_loss += metrics['loss']
                    train_epoch_acc += metrics['acc']

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
                        tb_writer.add_scalar('elapsed/train', elapsed_train, step)
                        tb_writer.add_scalar('Loss/train', iterval_loss, step)
                        tb_writer.add_scalar('Acc/train', interval_acc, step)
                        tb_writer.add_scalar('learning_rate', optim.param_groups[0]["lr"], step)
                        elapsed_train = 0
                        train_interval_loss = .0
                        train_interval_acc = .0

                    # save checkpoint and validate.
                    if step > 1 and step % self.config.step_save_ckpt == 0:
                        save_checkpoint(origin_model, os.path.join(self.model_dir, 'checkpoint.pt'),
                                        step, current_epoch, optim, scheduler)
                        model_files = save_model(origin_model, self.model_dir, model_files,
                                                self.config.keep_last_models, step)
                        get_logger().info('checkpoint saved at %d/%d', current_epoch, step)

                        if valid_dataloader is not None:
                            get_logger().info('%d/%d Start to validation', current_epoch, step)
                            model.eval()
                            valid_started = time.time()
                            with torch.no_grad():
                                metrics = self._valid_epoch(
                                    valid_dataloader, model, loss_fn)
                            elapsed_valid = time.time() - valid_started
                            tb_writer.add_scalar('elapsed/valid', elapsed_valid, step)
                            tb_writer.add_scalar('Loss/valid', metrics['loss'], step)
                            tb_writer.add_scalar('Acc/valid', metrics['acc'], step)
                            get_logger().info(
                                '%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                                current_epoch, step, round(metrics['loss'], 4), round(metrics['acc'], 4),
                                round(elapsed_valid, 2))

                            if train_dataloader is None:
                                break

                # if scheduler:
                #     scheduler.step(metrics=train_epoch_loss/iters)
                tb_writer.add_scalar('epoch', current_epoch, step)
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
            save_checkpoint(origin_model, os.path.join(self.model_dir, 'checkpoint.pt'),
                            step, current_epoch, optim, scheduler)

    def _load_vocab(self):
        with open(self.config.vocab_file, 'rt', encoding='utf8') as f:
            vocab = json.load(f)
        shutil.copy(self.config.vocab_file, self.model_dir)
        return vocab

    def fit(self):
        get_logger().info('Training started.')
        self._log_config()

        get_logger().info('Used device type: %s', self.device.type)

        origin_model = self._load_model()
        if self.config.compile_model:
            model = torch.compile(origin_model)
            get_logger().info('model compiled.')
        else:
            model = origin_model

        model.to(self.device)

        separator = '='*80
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

        # Expand GPU memory of model before load dataset.
        dummy_tensor = torch.randint(
            0, len(self.vocab)-1, [self.config.batch_size, self.config.seq_len], device=self.device)
        try:
            model(dummy_tensor)
            del dummy_tensor
        except torch.cuda.OutOfMemoryError as oom_exception:
            get_logger().error('CUDA out of memory before load dataset: %s', oom_exception)
            raise oom_exception

        get_logger().info('Loading dataset...')
        train_dataloader, valid_dataloader = self._load_datasets()
        datasets = ''
        if train_dataloader:
            iters = len(train_dataloader)
            datasets += f'trains: {len(train_dataloader.dataset)}, steps per epoch: {iters} '
        if valid_dataloader:
            datasets += f'valids: {len(valid_dataloader.dataset)}, '\
                f'steps per epoch: {len(valid_dataloader)}'
        get_logger().info('Dataset loaded. %s', datasets)

        try:
            self._train_loop(model, origin_model, train_dataloader, valid_dataloader)
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
