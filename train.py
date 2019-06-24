import argparse
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models

from utils import ProxyStaticLoss, ImageReader
from utils import create_id, get_transform, acc


##################################################
# step 1: Loading Data
##################################################
def loadData(meta_id):
    # balance data for each class
    TH = 300
    # append image
    data_dict_meta = {i: [] for i in range(max(meta_id) + 1)}
    for i, c in idx_to_ori_class.items():
        meta_class_id = meta_id[i]
        tra_imgs = train_data[c]
        if len(tra_imgs) > TH: tra_imgs = random.sample(tra_imgs, TH)
        data_dict_meta[meta_class_id] += tra_imgs

    data_transforms_tra = get_transform(DATA_NAME, 'train')
    data_transforms_val = get_transform(DATA_NAME, 'test')
    classSize = len(data_dict_meta)

    return data_dict_meta, data_transforms_tra, data_transforms_val, classSize


##################################################
# step 2: Set Model
##################################################
def setModel():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classSize)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model = model.cuda()
    optimizer = Adam(model.parameters())
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * NUM_EPOCH), int(0.8 * NUM_EPOCH)], gamma=0.01)
    return model, optimizer, lr_scheduler


##################################################
# step 3: Learning
##################################################
def optd(num_epochs, index):
    # recording epoch acc and best result
    best_acc = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
        lr_scheduler.step(epoch)
        tra_loss, tra_acc = tra()
        print('tra - Loss:{:.4f} - Acc:{:.4f}'.format(tra_loss, tra_acc))
        # deep copy the model
        if epoch >= 1 and tra_acc > best_acc:
            best_acc = tra_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, 'results/model_{:02}.pth'.format(index))
    print('Best tra acc: {:.2f}'.format(best_acc))
    return best_model


def tra():
    # Set model to training mode
    model.train()
    dsets = ImageReader(data_dict_meta, data_transforms_tra)
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    L_data, T_data, N_data = 0.0, 0, 0

    # iterate batch
    for data in dataLoader:
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            inputs_bt, labels_bt = data
            fvec = model(inputs_bt.cuda())
            loss = criterion(fvec, labels_bt)
            loss.backward()
            optimizer.step()

        _, preds_bt = torch.max(fvec.cpu(), 1)

        L_data += loss.item()
        T_data += torch.sum(preds_bt == labels_bt).item()
        N_data += len(labels_bt)

    return L_data / N_data, T_data / N_data


def eva(best_model, index):
    best_model.eval()
    dsets = ImageReader(test_data, data_transforms_val)
    dataLoader = torch.utils.data.DataLoader(dsets, BATCH_SIZE, shuffle=False, num_workers=8)

    Fvecs = []
    with torch.no_grad():
        for data in dataLoader:
            inputs_bt, labels_bt = data
            fvec = F.normalize(best_model(inputs_bt.cuda()), p=2, dim=1)
            Fvecs.append(fvec.cpu())

    Fvecs_all = torch.cat(Fvecs, 0)
    torch.save(dsets, 'results/testdsets.pth')
    torch.save(Fvecs_all, 'results/' + str(index) + 'testFvecs.pth')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop'], help='dataset name')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=12, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=12, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='meta class size')

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCH = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    ENSEMBLE_SIZE, META_CLASS_SIZE = opt.ensemble_size, opt.meta_class_size
    recall_ids = [int(k) for k in RECALLS.split(',')]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {'train_loss': []}
    for k in recall_ids:
        results['train_recall_{}'.format(k)], results['test_recall_{}'.format(k)] = [], []

    data_dict = torch.load('data/{}/data_dicts.pth'.format(DATA_NAME))
    print('Creating ID')
    ID = create_id(META_CLASS_SIZE, ENSEMBLE_SIZE, len(data_dict['train']))
    train_data, test_data = data_dict['train'], data_dict['test']
    # sort classes and fix the class order
    all_class = sorted(train_data)
    idx_to_ori_class = {i: all_class[i] for i in range(len(all_class))}

    for i in range(ID.size(1)):
        print('Training ensemble #{}'.format(i))
        meta_id = ID[:, i].tolist()
        data_dict_meta, data_transforms_tra, data_transforms_val, classSize = loadData(meta_id)
        model, optimizer, lr_scheduler = setModel()
        criterion = ProxyStaticLoss(classSize, classSize)
        best_model = optd(NUM_EPOCH, i)
        eva(best_model, i)
    acc('results/', ENSEMBLE_SIZE, recall_ids)
