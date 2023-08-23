import json
import requests
import torch
from torch.utils.data import DataLoader
from dataset_loader import mlm_dataset
import configuration
import logging
import tqdm
logging.getLogger().handlers.clear()

config_file = "/data/dhchoi/trained/sunny_side_up/v5k_bat128_d1024_h8_lyr6_ff2048_lr2e-4dcosinelr/config_ln_encoder.yaml"
save_file = 'outttt_n.tsv'


config = configuration.load_config_file(config_file)
with open(config.vocab_file, 'rt') as f:
    vocab = json.load(f)
id2token = {v: k for k, v in vocab.items()}
# train_dataloader = mlm_dataloader(
#     config.train_dataset_files, vocab, config.vocab_start_token_id,
#     config.seq_len, config.batch_size, config.shuffle_dataset_on_load, dynamic_masking=True
# )
dataset = mlm_dataset.MLMdatasetDynamic(
        config.train_dataset_files, vocab, config.vocab_start_token_id,
        config.seq_len, config.shuffle_dataset_on_load)#, config.train_sampling_ratio)
train_dataloader = DataLoader(
        dataset, config.batch_size, config.shuffle_dataset_on_load, num_workers=2,
        collate_fn=mlm_dataset.create_collate_fn(config.seq_len, vocab['__PAD__']),
        worker_init_fn=mlm_dataset.worker_init)
# valid_dataloader = mlm_dataloader(
#         config.valid_dataset_files, vocab, config.vocab_start_token_id,
#         config.seq_len, config.batch_size, dynamic_masking=False
#     )

f = open(save_file, 'wt', encoding='utf8')
data = ['input\tpredicted\tdetails\n']
try:
    for x, y in tqdm.tqdm(train_dataloader):
        for _x, _y in zip(x, y):
            input_seq = torch.where(_y > 0, _y, _x)
            input_seq = [id2token[token_id.item()] for token_id in input_seq]
            input_seq = list(filter(lambda x: x != '__PAD__', input_seq))
            for yy in _y.nonzero():
                input_seq[yy] = f'[{input_seq[yy.item()]}]'
            input_seq = ''.join(input_seq)
            input_seq = input_seq.replace('@@', ' ')
            resp = requests.post(f"http://localhost:8000/?text={_x.tolist()}", timeout=10)
            res = json.loads(resp.text)
            labels = _y[_y.nonzero(as_tuple=True)]
            ans = []
            for label in labels:
                ans.append(id2token[label.item()])
            row = f'{input_seq}'
            details = ''
            predicted = []
            for re in res:
                for r in re:
                    details += f'{r["text"]}\t{r["prob"]}\t'
                predicted.append(re[0]['text'])
            data.append(f'{row}\t{predicted}\t{details}\n'.replace('"', "''"))
            if len(data) >= 1000:
                f.writelines(data)
                data.clear()
except KeyboardInterrupt:
    pass

if data:
    f.writelines(data)
