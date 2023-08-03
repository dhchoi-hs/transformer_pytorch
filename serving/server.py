"""
Serving bert masked language model with ray serve lib.

** Do not run this file directly. Use below command in root project directory.
running command:
    $ CUDA_VISIBLE_DEVICES=0 serve run serving.server:generator --host 0
"""
import os
import ast
import json
from typing import List
import torch
from starlette.requests import Request
from ray import serve
import configuration
from models.lm_encoder import lm_encoder
import checkpoint
from serving.bpe_for_serving.bpe_codec_char import encode_bpe_char
from serving.bpe_for_serving.parse_excels import preprocess_text


@serve.deployment(
    ray_actor_options={'num_gpus': 1}
)
class PredictMasked:
    def __init__(self, _config: configuration.ConfigData, _model_file: str) -> None:
        # 1. load config & vocab
        with open(_config.vocab_file, 'rt', encoding='utf8') as f:
            self.vocab = json.load(f)
        self.padding_idx = self.vocab['__PAD__']
        self.id2token = {v: k for k, v in self.vocab.items()}

        # # 2. load model
        self.model = lm_encoder(
            _config.d_model, _config.h, _config.ff, _config.n_layers,
            len(self.vocab), padding_idx=self.padding_idx, activation='gelu')
        self.model = torch.compile(self.model)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # # # 3. load ckpt
        ckpt = checkpoint.load_ckpt(_model_file)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.train(False)
        self.max_seq_len = _config.seq_len

    def _collate(self, inputs):
        collated = []
        for inp in inputs:
            if len(inp) > self.max_seq_len:
                inp = inp[:self.max_seq_len]
            elif len(inp) == self.max_seq_len:
                pass
            else:
                inp = inp + [self.padding_idx]*(self.max_seq_len - len(inp))
            collated.append(inp)

        return collated

    @serve.batch(max_batch_size=4)
    @torch.no_grad()
    async def handle_batch(self, inputs: List[str]) -> List[str]:
        # 1. encode input texts.
        predict_mask = []
        for idx, inp in enumerate(inputs):
            try:
                tokens = ast.literal_eval(inp)
            except (ValueError, SyntaxError):
                tokens = encode_bpe_char(self.vocab, preprocess_text, inp)
            predict_mask.append(tokens)
            print(f'{idx+1}/{len(inputs)}', inp, [self.id2token[i] for i in tokens])
        predict_mask = self._collate(predict_mask)
        encoded = torch.LongTensor(predict_mask)

        # 2. get mask positions
        mask_indices = encoded == self.vocab['__MASK__']

        # 3. inference
        num_of_mask = (encoded != self.padding_idx).unsqueeze(-2).unsqueeze(-2)
        predict_a_seq = self.model(encoded.to('cuda'), num_of_mask.to('cuda'))
        predict_mask = torch.matmul(predict_a_seq[mask_indices], self.model.emb.table.T)
        predict_mask = torch.softmax(predict_mask, -1)

        # 4. decode output results of mask positions.
        val = iter(zip(*torch.topk(predict_mask, 3)))

        result = []
        for mask_index in mask_indices:
            masks = mask_index.nonzero()
            predicted_seq = []
            for _ in masks:
                predicted = []
                for prob, idx in zip(*next(val)):
                    predicted.append(
                        {
                            'text': self.id2token[idx.item()].replace('@@', ' '),
                            'prob': round(prob.item(), 3)
                        }
                    )
                predicted_seq.append(predicted)
            result.append(predicted_seq)

        return result

    async def __call__(self, request: Request) -> List[str]:
        return await self.handle_batch(request.query_params["text"])


MODEL_DIR = "/data/dhchoi/trained/sunny_side_up/acc97_full_newloader"

config = configuration.load_config_file(os.path.join(MODEL_DIR, 'config_ln_encoder.yaml'))
model_file = os.path.join(MODEL_DIR, 'checkpoint.pt')

generator = PredictMasked.bind(config, model_file)
