import random

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
            [transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return transform


class DiverseLoss(nn.Module):
    def __init__(self):
        super(DiverseLoss, self).__init__()

    def forward(self, output, positives, negatives, model):
        output = output.unsqueeze(dim=1)
        p_samples = positives.view(-1, *positives.size()[2:])
        n_samples = negatives.view(-1, *negatives.size()[2:])
        p_out = model(p_samples)
        n_out = model(n_samples)
        p_out = p_out.view(output.size(0), -1, p_out.size(-1))
        n_out = n_out.view(output.size(0), -1, n_out.size(-1))
        p_loss = torch.abs(output.norm(dim=-1) - p_out.norm(dim=-1)).mean(dim=-1)
        n_loss = torch.abs(output.norm(dim=-1) - n_out.norm(dim=-1)).mean(dim=-1).clamp(min=1e-8).pow(-1)
        p_direction_loss = (1 + F.cosine_similarity(output, p_out, dim=-1)).mean(dim=-1)
        loss = p_loss + n_loss + p_direction_loss
        return loss.mean()


class RecallMeter(meter.Meter):
    def __init__(self, topk=[1]):
        super(RecallMeter, self).__init__()
        self.topk = topk
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
    def __init__(self, data_name, data_type='train', k=10):

        if data_type == 'val':
            data = read_json('data/{}/train_images.json'.format(data_name))
        else:
            data = read_json('data/{}/{}_images.json'.format(data_name, data_type))
        self.transform = get_transform(data_name, data_type)
        self.data_type, self.k = data_type, k
        self.images, self.labels = list(data.keys()), list(data.values())
        # make map between classes and labels
        classes, labels = sorted(set(data.values())), {}
        if self.k > len(classes) - 1:
            raise IndexError('k must less than classes-1({})'.format(len(classes) - 1))
        for index, label in enumerate(classes):
            labels[label] = index
        for index, label in enumerate(self.labels):
            self.labels[index] = labels[label]
        # self.indexes = set(range(len(self.images)))
        self.dict_images = {}
        for img, label in zip(self.images, self.labels):
            if label in self.dict_images:
                self.dict_images[label].append(img)
            else:
                self.dict_images[label] = [img]

    def __getitem__(self, index):
        img_path, label = self.images[index], self.labels[index]
        img = self.transform(Image.open(img_path).convert('RGB'))

        if self.data_type == 'train':
            positive_database = self.dict_images[label]
            # make sure the search database don't contain itself
            positive_database = list(set(positive_database) - {img_path})
            # choose 1 positive
            positive_path = random.choice(positive_database)
            positive_img = self.transform(Image.open(positive_path).convert('RGB')).unsqueeze(dim=0)
            # select all negative samples
            negative_database, negative_imgs = [], []
            for database_label, database_images in self.dict_images.items():
                if database_label != label:
                    negative_database.append(database_images)
            # choose k negative samples
            negative_database = random.choices(negative_database, k=self.k)
            for images in negative_database:
                negative_path = random.choice(images)
                negative_img = self.transform(Image.open(negative_path).convert('RGB'))
                negative_imgs.append(negative_img)
            negative_imgs = torch.stack(negative_imgs)
            return img, positive_img, negative_imgs
        else:
            return img, label, index

    def __len__(self):
        return len(self.images)
