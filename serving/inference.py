import json
import requests
import torch
from dataset_loader.mlm_dataset import mlm_dataloader
import configuration


config_file = "/data/dhchoi/trained/sunny_side_up/v5k_bat128_d1024_h8_lyr6_ff2048_lr1e-4decay0.9_newloader/config_ln_encoder2.yaml"
save_file = 'outttt_tmp.tsv'


config = configuration.load_config_file(config_file)
with open(config.vocab_file, 'rt') as f:
    vocab = json.load(f)
id2token = {v: k for k, v in vocab.items()}
train_dataloader = mlm_dataloader(
    config.train_dataset_files, vocab, config.vocab_start_token_id,
    config.seq_len, config.batch_size, config.shuffle_dataset_on_load, dynamic_masking=True
)

# valid_dataloader = mlm_dataloader(
#         config.valid_dataset_files, vocab, config.vocab_start_token_id,
#         config.seq_len, config.batch_size, dynamic_masking=False
#     )

f = open(save_file, 'wt', encoding='utf8')
data = ['input\tpredicted\tdetails\n']
try:
    for x, y in train_dataloader:
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
                print(f'{len(train_dataloader)}')
except KeyboardInterrupt:
    pass

if data:
    f.writelines(data)
