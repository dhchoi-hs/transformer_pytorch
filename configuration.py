from dataclasses import dataclass, asdict
from typing import Union
import yaml


@dataclass()
class ConfigData:
    keep_last_models: int
    step_save_ckpt: int
    cuda_index: Union[int, None]
    compile_model: bool
    epoch: int
    batch_size: int
    learning_rate: float
    lr_scheduler: str
    lr_scheduler_kwargs: dict
    weight_decay: float
    d_model: int
    h: int
    ff: int
    n_layers: int
    p_dropout: float
    seq_len: int
    activation: str
    vocab_file: str
    vocab_start_token_id: int
    shuffle_dataset_on_load: bool
    train_dataset_files: list[str]
    train_sampling_ratio: float
    valid_dataset_files: list[str]
    valid_sampling_ratio: float


def load_config_file(config_file) -> ConfigData:
    with open(config_file, 'rt', encoding='utf8') as f:
        config = yaml.load(f, yaml.SafeLoader)

    return ConfigData(**config)


def validate_config(config: ConfigData):
    if config.keep_last_models < 1:
        raise ValueError('keep_last_models config value must be greater than 0.')


def convert_to_dict(config: ConfigData) -> dict:
    return asdict(config)
