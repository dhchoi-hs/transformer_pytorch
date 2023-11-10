import random
import torch
from torch.utils.data import Dataset
from hs_aiteam_pkgs.util.logger import get_logger
from dataset_loader.mlm_dataset import round_int
from collections import defaultdict

random.seed(7)


def collate_fn(samples, max_seq, padding_idx):
    collated_x = []
    collated_y = []
    for x, y in samples:
        if len(x) == max_seq:
            pass
        elif len(x) > max_seq:
            x = x[:max_seq]
        else:
            x = [*x, *[padding_idx]*(max_seq-len(x))]
        collated_x.append(x)
        collated_y.append(y)

    return torch.LongTensor(collated_x), torch.Tensor(collated_y)


def worker_init(worker_id):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    random.seed(7)

    return


class KFold:
    def __init__(self, k, dataset, shuffle) -> None:
        self.k = k
        self.split_train, self.split_valid = self._k_fold_split_dataset(dataset, shuffle)
    
    def _k_fold_split_dataset(self, dataset, shuffle):
        label_indices = defaultdict(list)

        for index, data in enumerate(dataset):
            label_indices[data[1][0]].append(index)

        if shuffle:
            for indices in label_indices.values():
                random.shuffle(indices)
        split_indices = defaultdict(list)
        for split in range(self.k):
            for label, indices in label_indices.items():
                split_indices[label].append(indices[round_int(len(indices)*split/self.k):round_int(len(indices)*(split+1)/self.k)])

        assert sum([sum(map(len, i)) for i in split_indices.values()]) == len(dataset), 'length different'

        splitted_train = []
        splitted_valid = []
        for split in range(self.k):
            train_indices = []
            valid_indices = []
            for indices in split_indices.values():
                trains = []
                trains.extend(indices[:split])
                trains.extend(indices[split+1:])
                for train in trains:
                    train_indices.extend(train)
                valid_indices.extend(indices[split])
            splitted_train.append(train_indices if shuffle else sorted(train_indices))
            splitted_valid.append(valid_indices if shuffle else sorted(valid_indices))

        return splitted_train, splitted_valid

    def __iter__(self, ):
        for st, sv in zip(self.split_train, self.split_valid):
            yield st, sv
    
    def __getitem__(self, i):
        return self.split_train[i], self.split_valid[i]


class TweetDisasterDataset(Dataset):
    def __init__(self, dataset_file: str, label_file, vocab, max_seq=None):
        super().__init__()
        self.dataset_file = dataset_file
        self.vocab = vocab

        with open(dataset_file, 'rt', encoding='utf8') as f:
            self.lines = f.readlines()
        self.lines = [list(map(int, line.strip().split(','))) for line in self.lines]
        with open(label_file, 'rt', encoding='utf8') as f:
            self.labels = f.readlines()
        self.labels = [[int(label)] for label in self.labels]

        if len(self.lines) != len(self.labels):
            raise IndexError('text and label length are different!')

        max_len = max([len(d) for d in self.lines])

        if max_len > max_seq:
            get_logger().warning(
                'dataset sequence length %d exceed config seq_len %d. '
                'dataset sentence exceeding seq_len will be truncated.',
                max_len, max_seq)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index], self.labels[index]
