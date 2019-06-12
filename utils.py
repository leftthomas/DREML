import numbers

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchnet.meter import meter
from torchvision import transforms

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


class RetrievalDataset(Dataset):
    def __init__(self, data_type, train=True):
        if train:
            self.annos = read_json(os.path.join(root_path, 'train.json'))
            self.transform = transform_train
        else:
            self.annos = read_json(os.path.join(root_path, 'test.json'))
            self.transform = transform_test
        self.imgs = []
        for img in self.annos:
            self.imgs.append(img)

    def __getitem__(self, index):
        img = self.imgs[index]
        name = img
        item = self.annos[img]
        img_path = item['path']
        label = item['label'] - 1
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, name

    def __len__(self):
        return len(self.imgs)


def load_data(data_type, batch_size=32):
    train_set = RetrievalDataset(data_type, train=True)
    test_set = RetrievalDataset(data_type, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8, shuffle=False)
    return train_loader, test_loader
