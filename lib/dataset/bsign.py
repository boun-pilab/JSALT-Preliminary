import os
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BSign22kDataset(Dataset):

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        if self.split == 'train':
            data_path = os.path.join(self.root, 'train.npy')
            label_path = os.path.join(self.root, 'train_labels.npy')
        elif self.split == 'val':
            data_path = os.path.join(self.root, 'val.npy')
            label_path = os.path.join(self.root, 'val_labels.npy')
        elif self.split == 'test':
            data_path = os.path.join(self.root, 'test.npy')
            label_path = os.path.join(self.root, 'test_labels.npy')
        else:
            raise ValueError('Invalid split')

        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def transform(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label



