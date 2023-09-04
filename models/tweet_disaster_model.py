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

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_pretrained_model()

    @staticmethod
    def _load_from_pretrain(model_file, _config):
        with open(_config.vocab_file, 'rt', encoding='utf8') as f:
            vocab = json.load(f)
        padding_idx = vocab['__PAD__']
        origin_model = lm_encoder(
            _config.d_model, _config.h, _config.ff, _config.n_layers,
            len(vocab), padding_idx=padding_idx, activation='gelu')
        model = torch.compile(origin_model)

        loaded_model = load_ckpt(model_file)
        model.load_state_dict(loaded_model)

        return origin_model


class TweetDisasterClassifierMLP(TweetDisasterClassifierBase):
    def __init__(self, pretrained_model: lm_encoder, seq_len, activation_function='relu') -> None:
        super().__init__(pretrained_model)

        # Classification layers
        self.ff1 = Linear(seq_len*self.pretrained_model.emb.d_model, 2048)
        self.activation = getattr(activation_functions, activation_function)
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
    def __init__(self, pretrained_model: lm_encoder, freeze_mode=1,
                 conv_filters=1, dropout_p=0.1) -> None:
        super().__init__(pretrained_model)

        self.cnns = nn.ModuleList([nn.Conv1d(
            self.pretrained_model.emb.d_model, conv_filters, kernel_size)
            for kernel_size in range(2, 6)])
        self.fc = Linear(len(self.cnns)*conv_filters, 1)
        self.dropout = Dropout.Dropout(dropout_p)
        self.freeze_mode = freeze_mode

    @classmethod
    def from_pretrained(cls, model_file, _config, freeze_mode=1, conv_filters=1, dropout_p=0.1):
        pretrained_model = cls._load_from_pretrain(model_file, _config)
        return cls(pretrained_model, freeze_mode, conv_filters, dropout_p)

    def train(self, mode: bool = True):
        super(TweetDisasterClassifierBase, self).train(mode)
        if self.freeze_mode == 1:
            self.freeze_pretrained_model()
        elif self.freeze_mode == 2:
            self.freeze_pretrained_model()
            self.pretrained_model.encoder.layers[-1].train(mode)
            self.pretrained_model.encoder.layer_norm.train(mode)
        elif self.freeze_mode == 3:
            pass
        else:
            raise ValueError('Unahndled freeze mode!')

    def _inference_pretrain1(self, x, mask=None):
        with torch.no_grad():
            _x = self.pretrained_model(x, mask)
        return _x

    def _inference_pretrain2(self, x, mask=None):
        with torch.no_grad():
            _x = self.pretrained_model.emb(x)
            _x = self.pretrained_model.pe(_x)
            for layer in self.pretrained_model.encoder.layers[:-1]:
                _x = layer(_x, mask)
        _x = self.pretrained_model.encoder.layers[-1](_x, mask)

        return self.pretrained_model.encoder.layer_norm(_x)

    def _inference_pretrain3(self, x, mask=None):
        return self.pretrained_model(x, mask)

    def forward(self, x, mask=None):
        _x = getattr(self, f'_inference_pretrain{self.freeze_mode}')(x, mask)

        outputs = []
        for input_sentence, sentence in zip(x, _x):
            sentence_without_padding = sentence[input_sentence != self.pretrained_model.padding_idx]
            sentence_without_padding = self.dropout(sentence_without_padding.transpose(-1, -2))
            conved_list = [cnn(sentence_without_padding)
                           for cnn in self.cnns]
            outputs.append(torch.cat(
                [activation_functions.relu(F.max_pool1d(conved, conved.size(-1))).
                 transpose(-1, -2).squeeze(-2) for conved in conved_list]))

        return self.fc(torch.stack(outputs)).sigmoid().squeeze(-1)


if __name__ == '__main__':
    torch.manual_seed(7)
    MODEL_FILE = '/data/dhchoi/trained/sunny_side_up/'\
        'v5k_bat128_d1024_h8_lyr6_ff2048_lr2e-4dcosinelr/model_670000.pt'
    CONFIG_FILE = '/data/dhchoi/trained/sunny_side_up/'\
        'v5k_bat128_d1024_h8_lyr6_ff2048_lr2e-4dcosinelr/config_ln_encoder.yaml'
    _config = load_config_file(CONFIG_FILE)

    model = TweetDisasterClassifierCNN.from_pretrained(MODEL_FILE, _config, 1, 3)
    model.train()
    num_params = 0
    for name, params in model.named_parameters():
        if params.requires_grad is True:
            num_params += params.numel()
    print(num_params)
    inp = torch.randint(2500, 5000, [2, _config.seq_len])
    inp[..., -20:] = model.pretrained_model.padding_idx
    logits = model(inp)
    print(logits)
    a = logits.mean().backward()
    gg = 123
