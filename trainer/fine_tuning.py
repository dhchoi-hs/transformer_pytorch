import os
import glob
import json
import time
import shutil
from functools import partial
from itertools import count
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import mlflow
from hs_aiteam_pkgs.util.logger import get_logger
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import tweet_disaster_dataset
import configuration
import configuration_fine_tuning
from checkpoint import load_ckpt, save_checkpoint, save_model
from training_iter import run_step_fine_tuning, run_epoch
from models.tweet_disaster_model import TweetDisasterClassifierCNN
from trainer.pre_train import PreTrainTrainer


class FineTuningTrainer(PreTrainTrainer):
    def __init__(self, _config_file, _pre_train_model_file, _pre_train_config_file,
                 _model_dir, _resume, memo) -> None:
        super().__init__(_config_file, _model_dir, _resume, memo)
        self.pre_train_model_file = _pre_train_model_file
        self.pre_train_config_file = _pre_train_config_file

        self.pre_train_config = None

    def _init_config(self):
        config = configuration_fine_tuning.load_config_file(self.config_path)
        configuration_fine_tuning.validate_config(config)
        return config

    def _init_pre_train_config(self):
        config = configuration.load_config_file(self.pre_train_config_file)
        configuration.validate_config(config)
        shutil.copy(self.pre_train_config_file, self.model_dir)
        return config

    def initialize_train(self):
        self.pre_train_config = self._init_pre_train_config()
        return super().initialize_train()

    def _load_model(self):
        # pretrained_model = mlflow.pytorch.load_model(
        #     'mlflow-artifacts:/514742599386093848/f73e2405fbe84386be38c90d4390d297/artifacts/best_val_acc_model')

        # model = TweetDisasterClassifierCNN(pretrained_model,
        #     self.config.unfreeze_last_layers, self.config.remove_last_layers,
        #     self.config.conv_filters, kernel_sizes=self.config.kernel_sizes,
        #     dropout_p=self.config.p_dropout)
        model = TweetDisasterClassifierCNN.from_pretrained(
            self.pre_train_model_file, self.pre_train_config, self.config.unfreeze_last_layers,
            self.config.remove_last_layers, self.config.conv_filters,
            kernel_sizes=self.config.kernel_sizes, dropout_p=self.config.p_dropout)

        return model

    def _load_vocab(self):
        with open(self.pre_train_config.vocab_file, 'rt', encoding='utf8') as f:
            vocab = json.load(f)
        return vocab

    def _load_datasets(self):
        collate_fn = partial(tweet_disaster_dataset.collate_fn,
                         max_seq=self.config.seq_len, padding_idx=self.vocab['__PAD__'])
        dataset = tweet_disaster_dataset.TweetDisasterDataset(
            self.config.train_dataset_file,
            self.config.train_dataset_label_file,
            self.vocab, self.config.seq_len)
        train_dataloader = DataLoader(
            dataset, self.config.batch_size, self.config.shuffle_dataset_on_load,
            collate_fn=collate_fn,)

        dataset = tweet_disaster_dataset.TweetDisasterDataset(
            self.config.valid_dataset_file,
            self.config.valid_dataset_label_file,
            self.vocab, self.config.seq_len)
        valid_dataloader = DataLoader(
            dataset, self.config.batch_size, self.config.shuffle_dataset_on_load,
            collate_fn=collate_fn,)

        return train_dataloader, valid_dataloader

    def _init_loss_fn(self):
        loss_fn = torch.nn.BCELoss()
        loss_fn.to(device=self.device)
        return loss_fn

    def _init_optimizer(self, model):
        optim = torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)
        return optim

    def _train_step(self, _data, model, criterion, optim=None):
        return run_step_fine_tuning(_data, model, criterion, optim, True, self.device)

    def _valid_epoch(self, dataset, model, criterion):
        return run_epoch(dataset, model, criterion, None, False, self.device, True)

    def _train_loop(self, model, origin_model, train_dataloader, valid_dataloader):
        '''
        model: model for training. It may or may not be compiled.
        origin_model: not compiled model. It's used to save and load state dict.
        '''
        loss_fn = self._init_loss_fn()
        optim = self._init_optimizer(model)
        scheduler = create_lr_scheduler(optim, self.config.lr_scheduler,
                                        **self.config.lr_scheduler_kwargs)

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

        train_interval_loss = train_interval_acc = train_interval_f1 = .0
        train_epoch_loss = train_epoch_acc = train_epoch_f1 = .0
        best_val_acc = best_val_f1 = 0
        logging_interval = 20
        elapsed_train = 0

        if step == 0:
            tb_writer.add_scalar('learning_rate', optim.param_groups[0]["lr"], step+1)
            mlflow.log_metric('learning_rate', optim.param_groups[0]["lr"], step+1)
        for current_epoch in count(start_epoch+1) \
                if self.config.epoch is None else range(start_epoch+1, self.config.epoch+1):
            tb_writer.add_scalar('epoch', current_epoch, step+1)
            mlflow.log_metric('epoch', current_epoch, step+1)
            for train_data in train_dataloader:
                step += 1
                model.train()
                training_started = time.time()
                metrics = self._train_step(train_data, model, loss_fn, optim)
                elapsed_train += time.time() - training_started
                train_interval_loss += metrics['loss']
                train_interval_acc += metrics['acc']
                train_interval_f1 += metrics['f1']
                train_epoch_loss += metrics['loss']
                train_epoch_acc += metrics['acc']
                train_epoch_f1 += metrics['f1']

                # change lr.
                if scheduler:
                    scheduler.step(step / iters)

                # logging per 20 steps.
                if step % logging_interval == 0:
                    interval_loss = train_interval_loss / logging_interval
                    interval_acc = train_interval_acc / logging_interval
                    interval_f1 = train_interval_f1 / logging_interval
                    get_logger().info(
                        '%d/%d training loss: %7.4f, acc: %7.4f, f1: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(interval_loss, 4),
                        round(interval_acc, 4), round(interval_f1, 4), round(elapsed_train,2))
                    tb_writer.add_scalar('elapsed/train', elapsed_train, step)
                    tb_writer.add_scalar('learning_rate', optim.param_groups[0]["lr"], step)
                    mlflow.log_metrics(
                        {
                            'elapsed/train': elapsed_train,
                            'learning_rate': optim.param_groups[0]["lr"],
                        },
                        step
                    )
                    elapsed_train = 0
                    train_interval_loss = .0
                    train_interval_acc = .0
                    train_interval_f1 = .0
            tb_writer.add_scalar('Loss/train', train_epoch_loss/len(train_dataloader), step)
            tb_writer.add_scalar('Acc/train', train_epoch_acc/len(train_dataloader), step)
            tb_writer.add_scalar('F1/train', train_epoch_f1/len(train_dataloader), step)
            mlflow.log_metrics(
                {
                    'Loss/train': train_epoch_loss/len(train_dataloader),
                    'Acc/train': train_epoch_acc/len(train_dataloader),
                    'F1/train': train_epoch_f1/len(train_dataloader),
                },
                step
            )
            train_epoch_loss = train_epoch_acc = train_epoch_f1 = .0

            # save checkpoint and validate.
            if step > 1:
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
                    tb_writer.add_scalar('F1/valid', metrics['f1'], step)
                    mlflow.log_metrics(
                        {
                            'elapsed/valid': elapsed_valid,
                            'Loss/valid': metrics['loss'],
                            'Acc/valid': metrics['acc'],
                            'F1/valid': metrics['f1'],
                        },
                        step
                    )
                    get_logger().info(
                        '%d/%d validation finished. loss: %7.4f, acc: %7.4f, f1: %7.4f, elapsed: %.2fs',
                        current_epoch, step, round(metrics['loss'], 4), round(metrics['acc'], 4),
                        round(metrics['f1'], 4), round(elapsed_valid, 2))

                    if best_val_acc < metrics['acc']:
                        best_val_acc = metrics['acc']
                        mlflow.pytorch.log_model(pytorch_model=origin_model, artifact_path='best_val_acc_model')
                        get_logger().info('best acc %f model is saved.', best_val_acc)
                    if best_val_f1 < metrics['f1']:
                        best_val_f1 = metrics['f1']
                        mlflow.pytorch.log_model(pytorch_model=origin_model, artifact_path='best_val_f1_model')
                        get_logger().info('best f1 %f model is saved.', best_val_f1)

                    if train_dataloader is None:
                        break

            tb_writer.add_scalar('epoch', current_epoch, step)
            mlflow.log_metric('epoch', current_epoch, step)
            get_logger().info('%s/%s Training a epoch finished.', current_epoch, step)

        save_checkpoint(origin_model, os.path.join(self.model_dir, 'checkpoint.pt'),
                        step, current_epoch, optim, scheduler)
