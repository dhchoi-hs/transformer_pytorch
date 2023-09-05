from dataclasses import dataclass, asdict
from typing import Union
import yaml


@dataclass()
class FineTuningConfigData:
    keep_last_models: int
    cuda_index: Union[int, None]
    compile_model: bool
    epoch: int
    freeze_mode: int
    batch_size: int
    learning_rate: float
    lr_scheduler: str
    lr_scheduler_kwargs: dict
    conv_filters: int
    kernel_sizes: list[int]
    weight_decay: float
    p_dropout: float
    seq_len: int
    shuffle_dataset_on_load: bool
    train_dataset_file: str
    train_dataset_label_file: str
    valid_dataset_file: str
    valid_dataset_label_file: str


def load_config_file(config_file) -> FineTuningConfigData:
    with open(config_file, 'rt', encoding='utf8') as f:
        config = yaml.load(f, yaml.SafeLoader)

    return FineTuningConfigData(**config)


def validate_config(config: FineTuningConfigData):
    if config.keep_last_models < 1:
        raise ValueError('keep_last_models config value must be greater than 0.')


def convert_to_dict(config: FineTuningConfigData) -> dict:
    return asdict(config)
