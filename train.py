import argparse
import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from utils import ProxyStaticLoss, ImageReader
from utils import create_id, get_transform, acc


class Learn():
    def __init__(self):

        self.train_data = data_dict['train']
        self.test_data = data_dict['test']

        self.init_lr = 0.01
        self.decay_rate = 0.01
        self.imgsize = 256

        # sort classes and fix the class order  
        all_class = sorted(self.train_data)
        self.idx_to_ori_class = {i: all_class[i] for i in range(len(all_class))}

    def run(self):
        for i in range(ID.size(1)):
            print('Training ensemble #{}'.format(i))
            self.l = i  # index of the ensembles
            self.meta_id = ID[:, i].tolist()
            self.decay_time = [False, False]
            self.loadData()
            self.setModel()
            self.criterion = ProxyStaticLoss(self.classSize, self.classSize)
            best_model = self.opt(NUM_EPOCH)
            self.eva(best_model)
        return

    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        # balance data for each class
        TH = 300

        # append image
        self.data_dict_meta = {i: [] for i in range(max(self.meta_id) + 1)}
        for i, c in self.idx_to_ori_class.items():
            meta_class_id = self.meta_id[i]
            tra_imgs = self.train_data[c]
            if len(tra_imgs) > TH: tra_imgs = random.sample(tra_imgs, TH)
            self.data_dict_meta[meta_class_id] += tra_imgs

        self.data_transforms_tra = get_transform(DATA_NAME, 'train')
        self.data_transforms_val = get_transform(DATA_NAME, 'test')

        self.classSize = len(self.data_dict_meta)
        print('output size: {}'.format(self.classSize))

        return

    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        print('Setting model')
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.classSize)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        return

    def lr_scheduler(self, epoch):
        if epoch >= 0.5 * NUM_EPOCH and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch >= 0.8 * NUM_EPOCH and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    ##################################################
    # step 3: Learning
    ##################################################
    def opt(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        best_epoch = 0
        best_acc = 0
        record = []
        for epoch in range(num_epochs):
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            self.lr_scheduler(epoch)

            tra_loss, tra_acc = self.tra()

            record.append((epoch, tra_loss, tra_acc))
            print('tra - Loss:{:.4f} - Acc:{:.4f}'.format(tra_loss, tra_acc))

            # deep copy the model
            if epoch >= 1 and tra_acc > best_acc:
                best_acc = tra_acc
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                torch.save(best_model, 'results/model_{:02}.pth'.format(self.l))

        torch.save(torch.Tensor(record), 'results/record_{:02}.pth'.format(self.l))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best tra acc: {:.2f}'.format(best_acc))
        print('Best tra acc in epoch: {}'.format(best_epoch))
        return best_model

    def tra(self):
        # Set model to training mode
        self.model.train()
        dsets = ImageReader(self.data_dict_meta, self.data_transforms_tra)
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=8)

        L_data, T_data, N_data = 0.0, 0, 0

        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data
                fvec = self.model(inputs_bt.cuda())
                loss = self.criterion(fvec, labels_bt)
                loss.backward()
                self.optimizer.step()

            _, preds_bt = torch.max(fvec.cpu(), 1)

            L_data += loss.item()
            T_data += torch.sum(preds_bt == labels_bt).item()
            N_data += len(labels_bt)

        return L_data / N_data, T_data / N_data

    def eva(self, best_model):
        best_model.eval()
        dsets = ImageReader(self.test_data, self.data_transforms_val)
        dataLoader = torch.utils.data.DataLoader(dsets, BATCH_SIZE, shuffle=False, num_workers=8)

        Fvecs = []
        with torch.set_grad_enabled(False):
            for data in dataLoader:
                inputs_bt, labels_bt = data
                fvec = F.normalize(best_model(inputs_bt.cuda()), p=2, dim=1)
                Fvecs.append(fvec.cpu())

        Fvecs_all = torch.cat(Fvecs, 0)
        torch.save(dsets, 'results/testdsets.pth')
        torch.save(Fvecs_all, 'results/' + str(self.l) + 'testFvecs.pth')
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

    x = Learn()
    x.run()
    acc('results/', ENSEMBLE_SIZE, recall_ids)
