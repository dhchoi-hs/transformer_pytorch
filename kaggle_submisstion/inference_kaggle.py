import argparse
import csv
import json
import re
import torch
from models.lm_encoder import lm_encoder
from models.tweet_disaster_model import TweetDisasterClassifierCNN
from configuration import load_config_file as load_pre_train_config_file
from configuration_fine_tuning import load_config_file
from hs_aiteam_pkgs.util.signal_handler import catch_kill_signal
from preprocess_tweetD import preprocess_text
from serving.bpe_for_serving.bpe_codec_char import encode_bpe_char

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
catch_kill_signal()


def load_model(pre_train_config, fine_tuning_config, ckpt_path, vocab_dict, device):
    model = TweetDisasterClassifierCNN(
        lm_encoder(
            pre_train_config.d_model,
            pre_train_config.h,
            pre_train_config.ff,
            pre_train_config.n_layers,
            len(vocab_dict),
            padding_idx=vocab_dict['__PAD__'],
            activation=pre_train_config.activation
            ),
        conv_filters=fine_tuning_config.conv_filters,
        kernel_sizes=fine_tuning_config.kernel_sizes
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def load_dataset(file):
    rows = []
    with open(file, encoding='utf8') as csv_f:
        reader = csv.reader(csv_f,)
        for a_row in reader:
            if a_row[0] == 'id':
                continue
            rows.append(a_row)

    return rows


def main(pre_train_config_file, fine_tuning_config_file, model_file):
    pre_train_config = load_pre_train_config_file(pre_train_config_file)
    fine_tuning_config = load_config_file(fine_tuning_config_file)

    with open(pre_train_config.vocab_file, 'rt', encoding='utf8') as f:
        vocab = json.load(f)
    device = torch.device('cuda:1')
    model = load_model(pre_train_config, fine_tuning_config, model_file, vocab, device)
    model.to(device)
    model.train(False)

    # LOAD DATASET
    dataset_rows = load_dataset('/data/corpus/tweet_disaster/test.csv')

    # PREPROCESSING & INFERENCE
    results = {}
    white_space = re.compile('%20')
    with torch.no_grad():
        for row in dataset_rows:
            _id, keyword, location, text = row
            if keyword:
                text = f'({white_space.sub(" ", keyword)}) {text}'

            tokens = encode_bpe_char(vocab, preprocess_text, text)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            output = model(tokens.unsqueeze(0))
            results[_id] = int(output.item()+0.5)

    with open('submission.csv', 'wt', encoding='utf8') as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['id', 'target'])
        csv_writer.writerows([[_id, text] for _id, text in results.items()])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--pre_trained_config', type=str,
                    default='output/fine_tuning_aihub_unfreeze1_no_keyword_no_mention/config_ln_encoder.yaml')
    ap.add_argument('-f', '--fine_tuning_config', type=str,
                    default='output/fine_tuning_aihub_unfreeze1_no_keyword_no_mention/config_fine_tuning.yaml')
    ap.add_argument('-m', '--model_file', type=str,
                    default='output/fine_tuning_aihub_unfreeze1_no_keyword_no_mention/model_1404.pt')

    args = ap.parse_args()

    main(args.pre_trained_config, args.fine_tuning_config, args.model_file)
