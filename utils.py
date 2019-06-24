import random

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module

RGBmean = {'CUB': [0.4707, 0.4601, 0.4549], 'CAR': [0.4853, 0.4965, 0.4295], 'SOP': [0.5807, 0.5396, 0.5044],
           'ISC': [0.8324, 0.8109, 0.8041]}
RGBstdv = {'CUB': [0.2767, 0.2760, 0.2850], 'CAR': [0.2237, 0.2193, 0.2568], 'SOP': [0.2901, 0.2974, 0.3095],
           'ISC': [0.2206, 0.2378, 0.2444]}


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
