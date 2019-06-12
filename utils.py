import numbers
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchnet.meter import meter
from torchvision import transforms


class RecallMeter(meter.Meter):
    def __init__(self, topk=[1], accuracy=False):
        super(RecallMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = np.atleast_1d(target.cpu().squeeze().numpy())
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        if np.ndim(output) == 1:
            output = output[np.newaxis]
        else:
            assert np.ndim(output) == 2, \
                'wrong output size (1D or 2D expected)'
            assert np.ndim(target) == 1, \
                'target and output do not match'
        assert target.shape[0] == output.shape[0], \
            'target and output do not match'
        topk = self.topk
        maxk = int(topk[-1])  # seems like Python3 wants int and not np.int64
        no = output.shape[0]

        pred = torch.from_numpy(output).topk(maxk, 1, True, True)[1].numpy()
        correct = pred == target[:, np.newaxis].repeat(pred.shape[1], 1)

        for k in topk:
            self.sum[k] += no - correct[:, 0:k].sum()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            assert k in self.sum.keys(), \
                'invalid k (this k was not provided at construction time)'
            if self.accuracy:
                return (1. - float(self.sum[k]) / self.n) * 100.0
            else:
                return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]


class ImageDataset(Dataset):
    def __init__(self, path, t):
        self.path = path
        self.T = t

    def __getitem__(self, index):
        sample, target = self.path[index], self.path[index]
        return self.T(sample), self.T(target)

    def __len__(self):
        return len(self.path)


def load_data(data_type, batch_size=32):
    train_set = ImageDataset('data/train', transforms.ToTensor())
    test_set = ImageDataset('data/test', transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


DB_dir = 'database'
DB_csv = 'data.csv'


class Database(object):

    def __init__(self):
        self._gen_csv()
        self.data = pd.read_csv(DB_csv)
        self.classes = set(self.data["cls"])

    def _gen_csv(self):
        if os.path.exists(DB_csv):
            return
        with open(DB_csv, 'w', encoding='UTF-8') as f:
            f.write("img,cls")
            for root, _, files in os.walk(DB_dir, topdown=False):
                cls = root.split('/')[-1]
                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, cls))

    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data
