import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from hs_aiteam_pkgs.util.logger import get_logger

random.seed(7)


def create_collate_fn(max_seq, padding_idx):
    def collate_fn(samples):
        collated_x = []
        collated_y = []
        # max_len = min(max(map(lambda x: len(x[0]), samples)), max_seq)
        # max_len = max([len(d[0]) for d in samples])
        # length = min(max_len, max_seq)
        for x, y in samples:
            if len(x) == max_seq:
                pass
            elif len(x) > max_seq:
                x = x[:max_seq]
                y = y[:max_seq]
            else:
                x = torch.cat([x, torch.LongTensor([padding_idx]*(max_seq-len(x)))])
                y = torch.cat([y, torch.LongTensor([0]*(max_seq-len(y)))])
            collated_x.append(x)
            collated_y.append(y)

        return torch.stack(collated_x), torch.stack(collated_y)

    return collate_fn


def worker_init(worker_id):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    random.seed(7)

    return


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
        seq = torch.LongTensor(seq)
        mask_len = int(length*0.15)
        mask_indices = torch.LongTensor(random.sample(range(0, length), mask_len))
        mask_indices_ = mask_indices[:int(mask_len*0.8)]
        # At least one mask token must be included in a sentence.
        if len(mask_indices_) == 0:
            mask_indices = torch.randint(0, length, [1])
            mask_indices_ = mask_indices.detach().clone()
            random_replace_token_indices = []
        else:
            random_replace_token_indices = torch.LongTensor(
                mask_indices[int(mask_len*0.8):int(mask_len*0.9)])
        masked_seq = seq.detach().clone()
        masked_seq[mask_indices_] = self.vocab['__MASK__']
        if len(random_replace_token_indices) > 0:
            masked_seq[random_replace_token_indices] = torch.LongTensor(
                random.sample(range(self.random_replaced_token_start_idx,
                    len(self.vocab)), len(random_replace_token_indices)))
        # label = torch.zeros(seq.shape).type(torch.long)
        label = torch.full_like(seq, -1).type(torch.long)
        label[mask_indices] = seq[mask_indices]

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
        del self.seqs

    def _set_dataset(self):
        return self._masking_a_seq()

    def _masking_a_seq(self):
        datas = []
        labels = []

        for seq in self.seqs:
            x, y = self.get_x_y(seq)
            datas.append(x)
            labels.append(y)

        # datas = torch.stack(datas)
        # labels = torch.stack(labels)

        return datas, labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def mlm_dataloader(dataset_files, vocab, start_index, max_sentence, batch_size,
                   shuffle=False, dynamic_masking=True):
    if dataset_files:
        if not dynamic_masking:
            dataset = MLMDatasetFixed(dataset_files, vocab, start_index, max_sentence, shuffle)
        else:
            dataset = MLMdatasetDynamic(dataset_files, vocab, start_index, max_sentence, shuffle)

        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=2,
                                collate_fn=create_collate_fn(max_sentence, vocab['__PAD__']),
                                worker_init_fn=worker_init)
        return dataloader

    return None
