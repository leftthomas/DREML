import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchnet.meter import meter
from torchvision import transforms

from data_utils import read_json

transform_train = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])


class RecallMeter(meter.Meter):
    def __init__(self, topk=[1], accuracy=False):
        super(RecallMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, index, label, database):
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


class DiverseLoss(nn.Module):
    def __init__(self):
        super(DiverseLoss, self).__init__()

    def forward(self, classes, positives, negatives, model):
        classes = classes.unsqueeze(dim=1)
        p_samples = positives.view(-1, *positives.size()[2:])
        n_samples = negatives.view(-1, *negatives.size()[2:])
        p_out, n_out = model(p_samples), model(n_samples)
        p_out = p_out.view(classes.size(0), -1, p_out.size(-1))
        n_out = n_out.view(classes.size(0), -1, n_out.size(-1))
        p_loss = torch.abs(classes.norm(dim=-1) - p_out.norm(dim=-1)).mean(dim=-1)
        n_loss = torch.abs(classes.norm(dim=-1) - n_out.norm(dim=-1)).mean(dim=-1).clamp(min=1e-8).pow(-1)
        p_direction_loss = (1 + F.cosine_similarity(classes, p_out, dim=-1)).mean(dim=-1)
        loss = p_loss + n_loss + p_direction_loss
        return loss.mean()


class RetrievalDataset(Dataset):
    def __init__(self, data_name, data_type='train', k=10):
        if data_type == 'train':
            data = read_json('data/{}/train.json'.format(data_name))
            self.transform = transform_train
        elif data_type == 'val':
            data = read_json('data/{}/train.json'.format(data_name))
            self.transform = transform_test
        else:
            data = read_json('data/{}/test.json'.format(data_name))
            self.transform = transform_test
        self.data_type = data_type
        self.images, self.labels = list(data.keys()), list(data.values())
        self.classes = read_json('data/{}/class.json'.format(data_name))
        self.k, self.indexes = k, set(range(len(self.images)))

    def __getitem__(self, index):
        img_path, label = self.images[index], self.labels[index]
        img = self.transform(Image.open(img_path).convert('RGB'))

        if self.data_type == 'train':
            positive_index = set(np.where(np.array(self.labels) == label)[0].tolist())
            negative_index = self.indexes - positive_index
            # make sure the search database don't contain itself
            positive_database = list(positive_index - {index})
            negative_database = list(negative_index)
            # choose k samples
            positive_database = random.choices(positive_database, k=self.k)
            negative_database = random.choices(negative_database, k=self.k)
            positives, negatives = [], []
            for i in range(self.k):
                positives.append(self.transform(Image.open(self.images[positive_database[i]]).convert('RGB')))
                negatives.append(self.transform(Image.open(self.images[negative_database[i]]).convert('RGB')))
            positives, negatives = torch.stack(positives), torch.stack(negatives)
            return img, positives, negatives
        else:
            return img, label, index

    def __len__(self):
        return len(self.images)

