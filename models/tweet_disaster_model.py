import json
import torch
from torch import nn
import torch.nn.functional as F
from models.lm_encoder import lm_encoder
from checkpoint import load_ckpt
from configuration import load_config_file
from model.utils.Linear import Linear
from model.utils import activation_functions, Dropout


class TweetDisasterClassifierBase(nn.Module):
    def __init__(self, pretrained_model: lm_encoder) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model

    def freeze_pretrained_model(self):
        self.pretrained_model.train(False)
        self.pretrained_model.requires_grad_(False)

    def forward(self, x, mask=None):
        raise NotImplementedError()

    @staticmethod
    def _load_from_pretrain(model_file, _config):
        with open(_config.vocab_file, 'rt', encoding='utf8') as f:
            vocab = json.load(f)
        padding_idx = vocab['__PAD__']
        origin_model = lm_encoder(
            _config.d_model, _config.h, _config.ff, _config.n_layers,
            len(vocab), padding_idx=padding_idx, activation='gelu')

        loaded_model = load_ckpt(model_file)

        # compile model if model state dict of ckpt was compiled.
        if next(iter(loaded_model.keys())).startswith('_orig_mod.'):
            model = torch.compile(origin_model)
        else:
            model = origin_model
        model.load_state_dict(loaded_model)

        return origin_model


class TweetDisasterClassifierMLP(TweetDisasterClassifierBase):
    def __init__(self, pretrained_model: lm_encoder, seq_len, activation_function='relu') -> None:
        super().__init__(pretrained_model)

        # Classification layers
        self.ff1 = Linear(seq_len*self.pretrained_model.emb.d_model, 2048)
        self.activation = getattr(activation_functions, activation_function)()
        self.dropout = Dropout.Dropout()
        self.ff2 = Linear(2048, 1)

    @classmethod
    def from_pretrained(cls, model_file, _config):
        pretrained_model = cls._load_from_pretrain(model_file, _config)
        return cls(pretrained_model, _config.seq_len)

    def forward(self, x, mask=None):
        with torch.no_grad():
            x = self.pretrained_model(x, mask)

        x = self.ff1(x.flatten(start_dim=-2))
        x = self.dropout(self.activation(x))
        x = self.ff2(x)
        return x.sigmoid().squeeze(-1)


class TweetDisasterClassifierCNN(TweetDisasterClassifierBase):
    def __init__(self, pretrained_model: lm_encoder, unfreeze_last_layers=1,
                 remove_last_layers=0, conv_filters=100, kernel_sizes=[3, 4, 5],
                 dropout_p=0.1) -> None:
        super().__init__(pretrained_model)

        self.cnns = nn.ModuleList([nn.Conv1d(
            self.pretrained_model.emb.d_model, conv_filters, kernel_size)
            for kernel_size in kernel_sizes])
        self.fc = Linear(len(self.cnns)*conv_filters, 1)
        self.dropout = Dropout.Dropout(dropout_p)
        self.unfreeze_last_layers = unfreeze_last_layers
        self.remove_last_layers = remove_last_layers
        self.max_kernel_size = max(kernel_sizes)
        self.cnn_activation_function = nn.ReLU()

        if remove_last_layers > 0:
            if remove_last_layers >= len(self.pretrained_model.encoder.layers):
                raise ValueError('size of removing last layers must be less than encoder layers.')
            self.pretrained_model.encoder.layers = \
                self.pretrained_model.encoder.layers[:-remove_last_layers]

    @classmethod
    def from_pretrained(cls, model_file, _config, unfreeze_last_layers=1,
                        remove_last_layers=0, conv_filters=100,
                        kernel_sizes=[3, 4, 5], dropout_p=0.1):
        pretrained_model = cls._load_from_pretrain(model_file, _config)
        return cls(pretrained_model, unfreeze_last_layers, remove_last_layers, 
                   conv_filters, kernel_sizes, dropout_p)

    def train(self, mode: bool = True):
        super(TweetDisasterClassifierBase, self).train(mode)
        self.freeze_pretrained_model()
        if mode and self.unfreeze_last_layers > 0:
            for layer in self.pretrained_model.encoder.layers[-self.unfreeze_last_layers:]:
                layer.requires_grad_(mode)
            if len(self.pretrained_model.encoder.layers) < self.unfreeze_last_layers:
                self.pretrained_model.encoder.layer_norm.requires_grad_(mode)
                self.pretrained_model.emb.requires_grad_(mode)

    def forward(self, x, mask=None):
        _x = self.pretrained_model(x, mask)
        outputs = []
        for input_sentence, sentence in zip(x, _x):
            # remove PAD tokens not needed for cnn.
            sentence_without_padding = sentence[input_sentence != self.pretrained_model.padding_idx]
            # sentence length must not be less than kernel size.
            if sentence_without_padding.size(0) < self.max_kernel_size:
                sentence_without_padding = F.pad(
                    sentence_without_padding,
                    (0, 0, 0, self.max_kernel_size - sentence_without_padding.size(0)))
            sentence_without_padding = sentence_without_padding.transpose(-1, -2)
            conved_list = [cnn(sentence_without_padding) for cnn in self.cnns]
            outputs.append(torch.cat(
                [self.cnn_activation_function(F.max_pool1d(conved, conved.size(-1))).
                 transpose(-1, -2).squeeze(-2) for conved in conved_list]))

        return self.fc(self.dropout(torch.stack(outputs))).sigmoid()


if __name__ == '__main__':
    torch.manual_seed(7)
    MODEL_FILE = '/data/dhchoi/trained/sunny_side_up/'\
        'v5k_bat128_d1024_h8_lyr6_ff2048_lr2e-4dcosinelr/model_670000.pt'
    CONFIG_FILE = '/data/dhchoi/trained/sunny_side_up/'\
        'v5k_bat128_d1024_h8_lyr6_ff2048_lr2e-4dcosinelr/config_ln_encoder.yaml'
    _config = load_config_file(CONFIG_FILE)

    model = TweetDisasterClassifierCNN.from_pretrained(
        MODEL_FILE, _config, 9, 1, 100, [3, 4, 5])
    origin_model = model
    # model = torch.compile(model)
    # model.to('cuda:0')
    model.train()
    num_params = 0
    for name, params in model.named_parameters():
        if params.requires_grad is True:
            num_params += params.numel()
    print(f'{num_params:,}')
    inp = torch.randint(2500, 5000, [128, 146])
    inp[..., -100:] = model.pretrained_model.padding_idx
    logits = model(inp)
    print(logits)
    a = logits.mean().backward()
    gg = 123
