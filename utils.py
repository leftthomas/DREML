import random

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

RGBmean, RGBstdv = {}, {}
# CUB
RGBmean['CUB'], RGBstdv['CUB'] = [0.4707, 0.4601, 0.4549], [0.2767, 0.2760, 0.2850]
# CAR
RGBmean['CAR'], RGBstdv['CAR'] = [0.4853, 0.4965, 0.4295], [0.2237, 0.2193, 0.2568]
# ICR
RGBmean['ISC'], RGBstdv['ISC'] = [0.8324, 0.8109, 0.8041], [0.2206, 0.2378, 0.2444]
# SOP
RGBmean['SOP'], RGBstdv['SOP'] = [0.5807, 0.5396, 0.5044], [0.2901, 0.2974, 0.3095]
# PKU
RGBmean['PKU'], RGBstdv['PKU'] = [0.3912, 0.4110, 0.4118], [0.2357, 0.2332, 0.2338]
# CIFAR100
RGBmean['CIFAR'], RGBstdv['CIFAR'] = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]


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

    if rank == None:
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


def recall2(Fvec_val, Fvec_gal, imgLab_val, imgLab_gal, rank=None):
    N = len(imgLab_val)
    imgLab_val = torch.LongTensor([imgLab_val[i] for i in range(len(imgLab_val))])
    imgLab_gal = torch.LongTensor([imgLab_gal[i] for i in range(len(imgLab_gal))])

    D = Fvec_val.mm(torch.t(Fvec_gal))

    if rank == None:
        _, idx = D.sort(1, descending=True)
        imgPre = imgLab_gal[idx[:, 0]]
        A = (imgPre == imgLab_val).float()
        return (torch.sum(A) / N).item()
    else:
        _, idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab_gal[idx[:, i]]
                A += (imgPre == imgLab_val).float()
            acc_list.append((torch.sum((A > 0).float()) / N).item())
        return acc_list


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
