import torch
from torch import nn
from hs_aiteam_pkgs.model.lr_scheduler import create_lr_scheduler
from model.Embeddings import Embeddings
from model.PositionalEncoding import PositionalEncoding
from model.Encoder import Encoder, EncoderLayer
from train_module import TrainModule


class lm_encoder(nn.Module):
    def __init__(self, d_model, h, ff, n_layers, n_vocabs, padding_idx=0, dropout_p=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.pe = PositionalEncoding(n_vocabs, d_model, dropout_p)
        self.emb = Embeddings(n_vocabs, d_model, padding_idx)
        self.encoder = Encoder(EncoderLayer(d_model, h, ff, dropout_p), n_layers)
    
    def forward(self, x, src_mask=None,):
        x = self.emb(x)
        x = self.pe(x)
        x = self.encoder(x, src_mask)

        return x


class TrainLM(TrainModule):
    def __init__(self, d_model, h, ff, n_layers, n_vocabs, padding_idx=0,
            dropout_p=0.1, device=None) -> None:
        super().__init__(lm_encoder(d_model, h, ff, n_layers, n_vocabs,
            padding_idx=padding_idx, dropout_p=dropout_p), device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion.to(device=self.device)
    
    def configure_optimizer(self, config):
        optim = torch.optim.Adam(self.model.parameters(),
            lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = create_lr_scheduler(optim, config.lr_scheduler,
                                        **config.lr_scheduler_kwargs)

        return optim, scheduler

    def train_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        pad_mask = (x != self.model.padding_idx).unsqueeze(-2).unsqueeze(-2)
        output = self.model(x, pad_mask)

        y_masked = y.bool()
        output_only_masked = output[y_masked]

        y_only_masked = y[y_masked]
        output_only_masked = torch.matmul(output_only_masked, self.model.emb.table.T)

        loss = self.criterion(output_only_masked, y_only_masked)

        output_labels = output_only_masked.argmax(dim=-1)
        a = torch.count_nonzero(output_labels == y_only_masked)
        acc = a.item() / y_only_masked.size(0)

        return loss, acc

    def validate_step(self, batch):
        return self.train_step(batch)
