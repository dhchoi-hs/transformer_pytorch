import random
import torch
from torch.utils.data import Dataset, DataLoader
from hs_aiteam_pkgs.util.logger import get_logger

random.seed(7)


def create_collate_fn(max_seq, padding_idx):
    def collate_fn(samples):
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

    return collate_fn


def worker_init(worker_id):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    random.seed(7)

    return


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
        self.labels = list(map(int, self.labels))

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
