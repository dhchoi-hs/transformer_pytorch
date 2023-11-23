import os
import glob
import time
from functools import partial
from itertools import count
from collections import defaultdict
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import mlflow
from hs_aiteam_pkgs.util.logger import get_logger
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from dataset_loader import tweet_disaster_dataset
from checkpoint import load_ckpt, save_checkpoint, save_model
from trainer.fine_tuning import FineTuningTrainer


class FineTuningCrossValidationTrainer(FineTuningTrainer):
    def __init__(self, _config_file, _pre_train_model_file, _finetuning_config_file,
                 _model_dir, _resume, memo) -> None:
        super().__init__(_config_file, _pre_train_model_file, _finetuning_config_file,
                         _model_dir, _resume, memo)
        self.n_fold = 5

    def _load_datasets(self):
        dataset = tweet_disaster_dataset.TweetDisasterDataset(
            self.config.train_dataset_file,
            self.config.train_dataset_label_file,
            self.vocab, self.config.seq_len)
        
        collate_fn = partial(tweet_disaster_dataset.collate_fn,
                         max_seq=self.config.seq_len, padding_idx=self.vocab['__PAD__'])
        train_dataloader = DataLoader(
            dataset, self.config.batch_size, self.config.shuffle_dataset_on_load,
            collate_fn=collate_fn,)

        return train_dataloader, None

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
            step = checkpoint['step']
        else:
            checkpoint = None
            step = 0

        tb_writer = SummaryWriter(
            self.model_dir, purge_step=None if not self.resume else step)

        k_fold = tweet_disaster_dataset.KFold(self.n_fold, train_dataloader.dataset,
                                              self.config.shuffle_dataset_on_load)
        collate_fn = partial(tweet_disaster_dataset.collate_fn,
            max_seq=self.config.seq_len, padding_idx=self.vocab['__PAD__'])
        
        fold_train_loss = defaultdict(list)
        fold_train_acc = defaultdict(list)
        fold_train_f1 = defaultdict(list)
        fold_valid_loss = defaultdict(list)
        fold_valid_acc = defaultdict(list)
        fold_valid_f1 = defaultdict(list)
        base_model_state_dict = deepcopy(origin_model.state_dict())

        for k, (train_indices, valid_indices) in enumerate(k_fold):
            if k > 0:
                origin_model.load_state_dict(base_model_state_dict)
                if self.config.compile_model:
                    model = torch.compile(origin_model)
                    get_logger().info('model compiled.')
                else:
                    model = origin_model
                optim = self._init_optimizer(model)

            k_fold_train_dataloader = DataLoader(
                Subset(train_dataloader.dataset, train_indices), self.config.batch_size,
                self.config.shuffle_dataset_on_load, collate_fn=collate_fn)
            k_fold_valid_dataloader = DataLoader(
                Subset(train_dataloader.dataset, valid_indices), self.config.batch_size,
                self.config.shuffle_dataset_on_load, collate_fn=collate_fn)

            iters = len(k_fold_train_dataloader)

            if self.resume:
                # checkpoint = load_ckpt(os.path.join(self.model_dir, 'checkpoint.pt'))
                # origin_model.load_state_dict(checkpoint['model_state_dict'])
                optim.load_state_dict(checkpoint['optimizer_state_dict'])

                # model_files = sorted(glob.glob(os.path.join(self.model_dir, 'model_*.pt')),
                #                     key=self._get_epoch_of_model_file)
                start_epoch = checkpoint['epoch']
                step = checkpoint['step']
                if scheduler and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # del checkpoint
            else:
                # model_files = []
                start_epoch = 0
                step = 0
            train_interval_loss = train_interval_acc = train_interval_f1 = .0
            logging_interval = 20
            elapsed_train = 0

            if k == 0 and step == 0:
                tb_writer.add_scalar('learning_rate', optim.param_groups[0]["lr"], step+1)
                mlflow.log_metric('learning_rate', optim.param_groups[0]["lr"], step+1)
            for current_epoch in count(start_epoch+1) \
                    if self.config.epoch is None else range(start_epoch+1, self.config.epoch+1):
                train_epoch_loss = 0
                train_epoch_acc = 0
                train_epoch_f1 = 0
                for train_data in k_fold_train_dataloader:
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
                        iterval_loss = train_interval_loss / logging_interval
                        interval_acc = train_interval_acc / logging_interval
                        interval_f1 = train_interval_f1 / logging_interval
                        get_logger().info(
                            '%d/%d (k-fold:%d/%d) training loss: %7.4f, acc: %7.4f, f1: %7.4f, elapsed: %.2fs',
                            current_epoch, step, k+1, self.n_fold, round(iterval_loss, 4),
                            round(interval_acc, 4), round(interval_f1, 4), round(elapsed_train,2))
                        if k == 0:
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
                fold_train_loss[k].append(train_epoch_loss/iters)
                fold_train_acc[k].append(train_epoch_acc/iters)
                fold_train_f1[k].append(train_epoch_f1/iters)
                train_epoch_loss = train_epoch_acc = train_epoch_f1 = .0

                # save checkpoint and validate.
                if step > 1:
                    # model_files = save_model(origin_model, self.model_dir, model_files,
                    #                         self.config.keep_last_models, f'{k}-{step}')
                    get_logger().info('checkpoint saved at %d/%d', current_epoch, step)

                    if k_fold_valid_dataloader is not None:
                        get_logger().info('%d/%d Start to validation', current_epoch, step)
                        model.eval()
                        valid_started = time.time()
                        with torch.no_grad():
                            metrics = self._valid_epoch(
                                k_fold_valid_dataloader, model, loss_fn)
                        elapsed_valid = time.time() - valid_started
                        if k == 0:
                            tb_writer.add_scalar('elapsed/valid', elapsed_valid, step)
                            mlflow.log_metric('elapsed/valid', elapsed_valid, step)
                        fold_valid_loss[k].append(metrics['loss'])
                        fold_valid_acc[k].append(metrics['acc'])
                        fold_valid_f1[k].append(metrics['f1'])
                        get_logger().info(
                            '%d/%d (k-fold:%d/%d) validation finished. loss: %7.4f, acc: %7.4f, f1: %7.4f, elapsed: %.2fs',
                            current_epoch, step, k+1, self.n_fold, round(metrics['loss'], 4), round(metrics['acc'], 4),
                            round(metrics['f1'], 4), round(elapsed_valid, 2))

                        if k_fold_train_dataloader is None:
                            break

                if k == 0:
                    tb_writer.add_scalar('epoch', current_epoch, step)
                get_logger().info('%s/%s Training a epoch finished.', current_epoch, step)

        self.write_metrics(tb_writer,
                           fold_train_loss,
                           fold_train_acc,
                           fold_train_f1,
                           fold_valid_loss,
                           fold_valid_acc,
                           fold_valid_f1)

    def write_metrics(self,
                      writer,
                      train_losses,
                      train_accs,
                      train_f1s,
                      valid_losses,
                      valid_accs,
                      valid_f1s):
        for i in range(self.config.epoch):
            avg_train_loss = 0
            avg_train_acc = 0
            avg_train_f1 = 0
            avg_valid_loss = 0
            avg_valid_acc = 0
            avg_valid_f1 = 0
            for n in range(self.n_fold):
                avg_train_loss += train_losses[n][i]
                avg_train_acc += train_accs[n][i]
                avg_train_f1 += train_f1s[n][i]
                avg_valid_loss += valid_losses[n][i]
                avg_valid_acc += valid_accs[n][i]
                avg_valid_f1 += valid_f1s[n][i]

            writer.add_scalar('K-fold average Loss/train', avg_train_loss/self.n_fold, i+1)
            writer.add_scalar('K-fold average Acc/train', avg_train_acc/self.n_fold, i+1)
            writer.add_scalar('K-fold average F1/train', avg_train_f1/self.n_fold, i+1)
            writer.add_scalar('K-fold average Loss/valid', avg_valid_loss/self.n_fold, i+1)
            writer.add_scalar('K-fold average Acc/valid', avg_valid_acc/self.n_fold, i+1)
            writer.add_scalar('K-fold average F1/valid', avg_valid_f1/self.n_fold, i+1)
            mlflow.log_metrics(
                {
                    'K-fold average Loss/train': avg_train_loss/self.n_fold,
                    'K-fold average Acc/train': avg_train_acc/self.n_fold,
                    'K-fold average F1/train': avg_train_f1/self.n_fold,
                    'K-fold average Loss/valid': avg_valid_loss/self.n_fold,
                    'K-fold average Acc/valid': avg_valid_acc/self.n_fold,
                    'K-fold average F1/valid': avg_valid_f1/self.n_fold,
                },
                i+1
            )
