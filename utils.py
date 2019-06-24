import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.module import Module
from torch.utils.data import Dataset

rgb_mean = {'car': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044]}
rgb_std = {'car': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095]}


def find_classes(data_dict):
    classes = [c for c in sorted(data_dict)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dict, class_to_idx):
    images = []
    idx_to_class = {}
    intervals = []
    i0, i1 = 0, 0

    for catg in sorted(dict):
        for fdir in dict[catg]:
            idx_to_class[i1] = class_to_idx[catg]
            images.append((fdir, class_to_idx[catg]))
            i1 += 1
        intervals.append((i0, i1))
        i0 = i1

    return images, intervals, idx_to_class


class ImageReader(Dataset):

    def __init__(self, data_dict, transform=None, target_transform=None):

        classes, class_to_idx = find_classes(data_dict)
        imgs, intervals, idx_to_class = make_dataset(data_dict, class_to_idx)

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images!"))

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx  # cat->1
        self.intervals = intervals
        self.idx_to_class = idx_to_class  # i(img idx)->2(class)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def createID(num_int, Len, N):
    """uniformly distributed"""
    multiple = N // num_int
    remain = N % num_int
    if remain != 0: multiple += 1

    ID = torch.zeros(N, Len)
    for i in range(Len):
        idx_all = []
        for _ in range(multiple):
            idx_base = [j for j in range(num_int)]
            random.shuffle(idx_base)
            idx_all += idx_base

        idx_all = idx_all[:N]
        random.shuffle(idx_all)
        ID[:, i] = torch.Tensor(idx_all)

    return ID.long()


def recall(Fvec, imgLab, rank=None):
    # Fvec: torch.Tensor. N by dim feature vector
    # imgLab: a list. N related labels list
    # rank: a list. input k(R@k) you want to calcualte 
    N = len(imgLab)
    imgLab = torch.LongTensor([imgLab[i] for i in range(len(imgLab))])

    D = Fvec.mm(torch.t(Fvec))
    D[torch.eye(len(imgLab)).byte()] = -1

    if rank is None:
        _, idx = D.sort(1, descending=True)
        imgPre = imgLab[idx[:, 0]]
        A = (imgPre == imgLab).float()
        return (torch.sum(A) / N).item()
    else:
        _, idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab[idx[:, i]]
                A += (imgPre == imgLab).float()
            acc_list.append((torch.sum((A > 0).float()) / N).item())
        return torch.Tensor(acc_list)


class ProxyStaticLoss(Module):
    def __init__(self, embed_size, proxy_num):
        """one proxy per class"""
        super(ProxyStaticLoss, self).__init__()
        self.proxy = torch.eye(proxy_num).cuda()

    def forward(self, fvec, fLvec):
        N = fLvec.size(0)

        # distance matrix
        Dist = fvec.mm((self.proxy).t())

        # loss
        Dist = -F.log_softmax(Dist, dim=1)
        loss = Dist[torch.arange(N), fLvec].mean()
        print('loss:{:.4f}'.format(loss.item()), end='\r')

        return loss
