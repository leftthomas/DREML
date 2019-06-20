import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchnet.meter import meter
from torchvision import transforms

from data_utils import read_json

rgb_mean = {'cars': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044]}
rgb_std = {'cars': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095]}


def get_transform(data_name, data_type):
    normalize = transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])
    if data_type == 'train':
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.RandomCrop(256), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), normalize])
    return transform


class DiverseLoss(nn.Module):
    def __init__(self):
        super(DiverseLoss, self).__init__()

    def forward(self, output, target):
        batch_size = output.size(0)
        # loss
        dist = -F.log_softmax(output, dim=-1)
        loss = dist[torch.arange(batch_size), target].mean()
        return loss


class RecallMeter(meter.Meter):
    def __init__(self, topk=[1]):
        super(RecallMeter, self).__init__()
        self.topk = np.sort(topk)
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, index, label, database, database_labels):
        no = output.shape[0]
        output, index, label = output.unsqueeze(dim=1), index.unsqueeze(dim=-1), label.unsqueeze(dim=-1)
        database = database.unsqueeze(dim=0)

        pred = torch.argsort(torch.abs(output.norm(dim=-1) - database.norm(dim=-1)))
        # make sure it don't contain itself
        pred = pred[pred != index].view(no, -1)
        for k in self.topk:
            recalled = pred[:, 0:k]
            correct = (database_labels[recalled] == label).any(dim=-1)
            self.sum[k] += correct.sum().item()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]


class RetrievalDataset(Dataset):
    def __init__(self, data_name, data_type='train'):

        data = read_json('data/{}/{}_images.json'.format(data_name, data_type))
        self.transform = get_transform(data_name, data_type)
        self.images, self.labels = list(data.keys()), list(data.values())
        self.classes, labels = sorted(set(data.values())), {}
        for index, label in enumerate(self.classes):
            labels[label] = index
        for index, label in enumerate(self.labels):
            self.labels[index] = labels[label]

    def __getitem__(self, index):
        img_path, label = self.images[index], self.labels[index]
        img = self.transform(Image.open(img_path).convert('RGB'))
        return img, label

    def __len__(self):
        return len(self.images)
