import torch

from utils import recall


def acc(src, L):
    # src: result directory
    # L : total ensembled size

    # loading dataset info
    dsets = torch.load(src + 'testdsets.pth')

    # loading feature vectors
    R = [torch.load(src + str(d) + 'testFvecs.pth') for d in range(L)]
    R = torch.cat(R, 1)
    print(R.size())

    acc_list = recall(R, dsets.idx_to_class, rank=[1, 2])

    print('R@1:{:.2f} R@2:{:.2f}'.format(acc_list[0].item() * 100, acc_list[1].item() * 100))
