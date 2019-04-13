import torch.nn.functional as F
from torch import nn
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

