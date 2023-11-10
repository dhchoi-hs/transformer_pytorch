import os
import torch
from hs_aiteam_pkgs.util.logger import get_logger


def load_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    return checkpoint


def save_model(_model, _model_dir, model_files, keep_last_models, epoch):
    model_file = os.path.join(_model_dir, f'model_{epoch}.pt')
    torch.save(_model.state_dict(), model_file)
    model_files.append(model_file)
    if len(model_files) > keep_last_models:
        for model_file in model_files[:-keep_last_models]:
            try:
                os.remove(model_file)
            except Exception as e:
                get_logger().warning('Deleting model file fails. %s, %s', model_file, e)
        model_files = model_files[-keep_last_models:]

    return model_files


def save_checkpoint(model, filename, step, current_epoch, optim, scheduler=None):
    save_data = {
            'step': step,
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }
    if scheduler:
        save_data['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(save_data, filename)
