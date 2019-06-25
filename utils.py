import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044]}
rgb_std = {'car': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095]}


def get_transform(data_name, data_type):
    normalize = transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])
    if data_type == 'train':
        transform = transforms.Compose([transforms.Resize(int(256 * 1.1)), transforms.RandomCrop(256),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), normalize])
    return transform


# random assign meta class for all classes
def create_id(meta_class_size, num_class):
    multiple = num_class // meta_class_size
    remain = num_class % meta_class_size
    if remain != 0:
        multiple += 1

    idx_all = []
    for _ in range(multiple):
        idx_base = [j for j in range(meta_class_size)]
        random.shuffle(idx_base)
        idx_all += idx_base

    idx_all = idx_all[:num_class]
    random.shuffle(idx_all)
    return idx_all


# according meta class to load data
def load_data(meta_id, data_dict):
    # balance data for each class
    num_samples, meta_data_dict = 300, {}
    for index, label in enumerate(sorted(data_dict)):
        meta_class, image_list = meta_id[index], data_dict[label]
        if len(image_list) > num_samples:
            image_list = random.sample(image_list, num_samples)
        if meta_class in meta_data_dict:
            meta_data_dict[meta_class] += image_list
        else:
            meta_data_dict[meta_class] = image_list
    return meta_data_dict


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


def acc(src, L, recall_ids):
    # src: result directory
    # L : total ensembled size

    # loading dataset info
    dsets = torch.load(src + 'testdsets.pth')

    # loading feature vectors
    R = [torch.load(src + str(d) + 'testFvecs.pth') for d in range(L)]
    R = torch.cat(R, 1)
    print(R.size())

    acc_list = recall(R, dsets.idx_to_class, rank=recall_ids)

    desc = ''
    for index, id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f} '.format(id, acc_list[index].item() * 100)
    print(desc)
