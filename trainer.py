from torch import no_grad
from torch.utils.data import DataLoader
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from hs_aiteam_pkgs.util.logger import get_logger
from hs_aiteam_pkgs.util.signal_handler import SigTermException
from hs_aiteam_pkgs.util.time_logger import TimeLogger
from checkpoint import load_ckpt, save_checkpoint, save_model
from train_module import TrainModule


class Trainer:
    def __init__(self, model_dir, ) -> None:
        self.model_dir = model_dir
        self.sw = SummaryWriter(os.path.join(model_dir, 'log'))
        self.logging_interval = 20

    def fit(self, train_module: TrainModule, dataloader: DataLoader, epochs,
            valid_dataloader: DataLoader = None):
        if train_module.model is None:
            raise ValueError('model is None')

        optims = train_module.configure_optimizer()

        if isinstance(optims, (list, tuple)) and len(optims) > 1:
            optim = optims[0]
            scheduler = optims[1]
        else:
            optim = optims
            scheduler = None

        elapsed_train = 0
        step = current_epoch = 1
        train_interval_loss = 0
        train_interval_acc = 0
        t = TimeLogger()
        vt = TimeLogger()
        
        try:
            for i in range(epochs):
                train_module.model.train()
                for d in dataloader:
                    t.reset_start_time()
                    optim.zero_grad()
                    loss = train_module.train_step(d)
                    #############
                    # acc
                    #############
                    loss.backward()
                    optim.step()
                    loss = loss.item()
                    train_module.backward(loss)
                    train_interval_loss += loss
                    if scheduler:
                        scheduler.step()
                    elapsed_train += t.get_elapsed_time_seconds()
                    
                    # logging per 20 steps.
                    if step % self.logging_interval == 0:
                        iterval_loss = train_interval_loss / self.logging_interval
                        interval_acc = train_interval_acc / self.logging_interval
                        get_logger().info('%d/%d training loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                            current_epoch, step, round(iterval_loss, 4),
                            round(interval_acc, 4), round(elapsed_train, 2))
                        self.sw.add_scalar('elapsed/train', elapsed_train, step)
                        self.sw.add_scalar('Loss/train', iterval_loss, step)
                        self.sw.add_scalar('Acc/train', interval_acc, step)
                        self.sw.add_scalar('learning_rate', optim.param_groups[0]["lr"], step)
                        elapsed_train = 0
                        train_interval_loss = .0
                        train_interval_acc = .0
                    step += 1

                if valid_dataloader:
                    train_module.model.train(False)
                    val_loss = 0
                    vt.reset_start_time()
                    with no_grad():
                        for d in valid_dataloader:
                            loss = train_module.validate_step(d)
                            if isinstance(loss, torch.tensor):
                                loss = loss.item()
                            val_loss += loss
                    elapsed_valid = vt.get_elapsed_time_seconds()
                    self.sw.add_scalar('elapsed/valid', elapsed_valid, step)
                    self.sw.add_scalar('Loss/valid', val_loss/len(valid_dataloader), step)
                    # self.sw.add_scalar('Acc/valid', val_acc, step)
                    get_logger().info('%d/%d validation finished. loss: %7.4f, acc: %7.4f, elapsed: %.2fs',
                                        current_epoch, step, round(val_loss, 4), round(val_acc, 4), round(elapsed_valid, 2))
                            
                current_epoch += 1
        except KeyboardInterrupt:
            get_logger().info('Training stopped by Ctrl+C.')
            save_checkpoint(train_module.model, os.path.join(
                self.model_dir, 'checkpoint.pt'), step, current_epoch, optim, scheduler)
        except SigTermException:
            get_logger().info('Training stopped by sigterm.')
            save_checkpoint(train_module.model, os.path.join(
                self.model_dir, 'checkpoint.pt'), step, current_epoch, optim, scheduler)
        except torch.cuda.OutOfMemoryError as oom_exception:
            get_logger().error('CUDA out of memory. :%s', oom_exception)
        except Exception as exception:
            get_logger().error('Exception occured during training. %s', exception)
        else:
            get_logger().info("All training finished.")
