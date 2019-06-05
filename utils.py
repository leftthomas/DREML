import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, path, t):
        self.path = path
        self.T = t

    def __getitem__(self, index):
        sample, target = self.path[index], self.path[index]
        return self.T(sample), self.T(target)

    def __len__(self):
        return len(self.path)


def load_data(batch_size=64):
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
