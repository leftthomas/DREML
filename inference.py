import argparse

import torch

from utils import recall


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='cars', type=str, choices=['cars', 'cub', 'sop'], help='dataset name')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--ensembled_size', default=12, type=int, help='test ensembled size')

    opt = parser.parse_args()
    RECALLS, ENSEMBLED_SIZE = opt.recalls, opt.ensembled_size
    recall_ids = [int(k) for k in RECALLS.split(',')]
    acc('results/', ENSEMBLED_SIZE, recall_ids)
