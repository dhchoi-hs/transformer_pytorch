import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from hs_aiteam_pkgs.util.logger import get_logger


random.seed(7)


class MLMdatasetDynamic(Dataset):
    def __init__(self, dataset_files: list, vocab, start_index, max_sentence,
                 shuffle=False, sampling_ratio=1.):
        super().__init__()
        self.dataset_files = dataset_files
        self.vocab = vocab
        self.random_replaced_token_start_idx = start_index
        self.max_sentence = max_sentence
        self.shuffle = shuffle
        self.sampling_ratio = sampling_ratio
        self.seqs = self.load_dataset()

    def load_dataset(self):
        datasets = []
        for dataset_file in self.dataset_files:
            with open(dataset_file, 'rt') as f:
                datasets.append([list(map(int, r.strip().split(','))) for r in f.readlines()])

        max_len = max([max(map(len, d)) for d in datasets])

        if max_len > self.max_sentence:
            get_logger().warning(
                'dataset sequence length %d exceed config seq_len %d. '
                'dataset sentence exceeding seq_len will be truncated.',
                max_len, self.max_sentence)

        seqs = []
        for dataset in datasets:
            if self.shuffle:
                random.shuffle(dataset)
            for _seq in tqdm(dataset[:int(len(dataset)*self.sampling_ratio)]):
                seqs.append(_seq[:self.max_sentence])

        if self.shuffle:
            random.shuffle(seqs)

        return seqs

    def get_x_y(self, seq):
        length = len(seq)
        seq = torch.LongTensor(seq + [self.vocab['__PAD__']]*(self.max_sentence-length))
        mask_len = int(length*0.15)
        mask_indices = torch.randint(0, length, [mask_len])
        mask_indices_ = mask_indices[:int(mask_len*0.8)]
        # At least one mask token must be added in a sentence.
        if len(mask_indices_) == 0:
            mask_indices_ = torch.randint(0, length, [1])
            random_replace_token_indices = []
        else:
            random_replace_token_indices = torch.LongTensor(
                mask_indices[int(mask_len*0.8):int(mask_len*0.9)])
        mask = torch.zeros(seq.shape).bool()
        mask[mask_indices] = True
        label = torch.where(mask, seq, 0)
        masked_seq = seq.clone()
        masked_seq[mask] = self.vocab['__MASK__']
        masked_seq[random_replace_token_indices] = torch.randint(
            self.random_replaced_token_start_idx, len(self.vocab),
            [len(random_replace_token_indices)])

        return masked_seq, label

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.get_x_y(self.seqs[index])


class MLMDatasetFixed(MLMdatasetDynamic):
    def __init__(self, dataset_files: list, vocab, start_index, max_sentence,
                 shuffle=False, sampling_ratio=1.0):
        super().__init__(dataset_files, vocab, start_index, max_sentence,
                         shuffle, sampling_ratio)

        self.x, self.y = self._set_dataset()
        assert len(self.x) == len(self.y)

    def _set_dataset(self):
        datas = []
        labels = []

        for seq in self.seqs:
            x, y = self.get_x_y(seq)
            datas.append(x)
            labels.append(y)

        datas = torch.stack(datas)
        labels = torch.stack(labels)

        return datas, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.get_x_y(self.seqs[index])


def mlm_dataloader(dataset_files, vocab, start_index, max_sentence, batch_size,
                   shuffle=False, dynamic_masking=True):
    if dataset_files:
        if not dynamic_masking:
            dataset = MLMDatasetFixed(dataset_files, vocab, start_index, max_sentence, shuffle)
        else:
            dataset = MLMdatasetDynamic(dataset_files, vocab, start_index, max_sentence, shuffle)

        dataloader = DataLoader(dataset, batch_size)
        return dataloader

    return None
