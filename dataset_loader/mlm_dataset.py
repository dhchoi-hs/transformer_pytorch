import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(7)


class MLMdataset(Dataset):
    def __init__(self, dataset_files, vocab, start_index, max_sentence):
        super().__init__()
        print('Loading dataset...')
        self.dataset = []
        for dataset_file in dataset_files:
            with open(dataset_file, 'rt') as f:
                self.dataset.extend(f.readlines())
        self.dataset = [list(map(int, r.strip().split(','))) for r in self.dataset]
        max_len = max(map(len, self.dataset))

        assert max_len <= max_sentence, f'max_len: {max_len}, max_sentence: {max_sentence}'
        self.max_sentence = max_sentence

        self.vocab = vocab

        self.SYMBOL_INDEX = start_index

        self.x, self.y = self.prepare_mlm_new()
        assert len(self.x) == len(self.y)
    
    def prepare_mlm1(self):
        """
        Not used function. replaced to prepare_mlm_new().
        keep this function to record using masking in different ways
        """
        x = self.dataset_.clone()

        entire_mask = (torch.zeros_like(x).float().uniform_() >= 0.85) & (x >= self.SYMBOL_INDEX)

        # 80% to MASK
        mask_ = torch.bernoulli(torch.fill(entire_mask.float(), 0.8)).bool() & entire_mask
        masked_x = torch.masked_fill(x, mask_, self.vocab['__MASK__'])
        
        # 10% to random token
        mask_random = torch.bernoulli(torch.fill(entire_mask.float(), 0.5)).bool() & entire_mask & ~mask_
        inp = torch.randint(self.SYMBOL_INDEX, len(vocab), masked_x.shape)
        masked_x = torch.where(mask_random, inp, masked_x)

        return masked_x, x
    
    def prepare_mlm_new(self):
        seqs = []
        labels = []
        for _seq in tqdm(self.dataset[:12800]):
            length = len(_seq)
            seq = torch.LongTensor(_seq + [self.vocab['__PAD__']]*(self.max_sentence-length))
            mask_len = int(length*0.15)
            mask_indices = torch.randint(0, length, [mask_len])
            mask_indices_ = mask_indices[:int(mask_len*0.8)]
            # At least one mask token must be added in a sentence.
            if len(mask_indices_) == 0:
                mask_indices_ = torch.randint(0, length, [1])
                random_replace_token_indices = []
            else:
                random_replace_token_indices = torch.LongTensor(mask_indices[int(mask_len*0.8):int(mask_len*0.9)])
            mask = torch.zeros(seq.shape).bool()
            mask[mask_indices_] = True
            label = torch.where(mask, seq, 0)
            masked_seq = seq.clone()
            masked_seq[mask] = self.vocab['__MASK__']
            masked_seq[random_replace_token_indices] = torch.randint(self.SYMBOL_INDEX, len(self.vocab), [len(random_replace_token_indices)])
            seqs.append(masked_seq)
            labels.append(label)
        
        x = torch.stack(seqs)
        y = torch.stack(labels)

        return x, y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def mlm_dataloader(dataset_files, vocab, start_index, max_sentence, batch_size):
    if dataset_files:
        dataset = MLMdataset(dataset_files, vocab, start_index, max_sentence)
        dataloader = DataLoader(dataset, batch_size)
        return dataloader
    else:
        return None
