import argparse
import copy
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader

from model import Model
from utils import ProxyStaticLoss, ImageReader
from utils import create_id, get_transform, acc, load_data


def train(net, meta_data, optimizer):
    net.train()
    data_set = ImageReader(meta_data, get_transform(DATA_NAME, 'train'))
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    l_data, t_data, n_data = 0.0, 0, 0
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        out = net(inputs.to(DEVICE))
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(out, 1)
        l_data += loss.item()
        t_data += torch.sum(pred.cpu() == labels).item()
        n_data += len(labels)

    return l_data / n_data, t_data / n_data


def eval(net, ensemble_num):
    net.eval()
    data_set = ImageReader(test_data, get_transform(DATA_NAME, 'test'))
    data_loader = DataLoader(data_set, BATCH_SIZE, shuffle=False, num_workers=8)

    features = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            out = net(inputs.to(DEVICE))
            out = F.normalize(out)
            features.append(out.cpu())
    features = torch.cat(features, 0)
    if not os.path.exists('epochs/test_dataset.pth'):
        torch.save(data_set, 'epochs/test_dataset.pth')
    torch.save(features, 'epochs/test_features_{:02}.pth'.format(ensemble_num))


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
    train_data, test_data = data_dict['train'], data_dict['test']

    # sort classes and fix the class order
    all_class = sorted(train_data)
    idx_to_ori_class = {i: all_class[i] for i in range(len(all_class))}

    for i in range(1, ENSEMBLE_SIZE + 1):
        print('Training ensemble #{}'.format(i))
        meta_id = create_id(META_CLASS_SIZE, len(data_dict['train']))
        meta_data_dict, classSize = load_data(meta_id, idx_to_ori_class, train_data)
        model = Model(classSize).to(DEVICE)
        optimizer = Adam(model.parameters())
        lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * NUM_EPOCH), int(0.8 * NUM_EPOCH)], gamma=0.01)
        criterion = ProxyStaticLoss()
        # recording epoch acc and best result
        best_acc = 0
        for epoch in range(1, NUM_EPOCH + 1):
            lr_scheduler.step()
            tra_loss, tra_acc = train(model, meta_data_dict, optimizer)
            print('Epoch {}/{} - Loss:{:.4f} - Acc:{:.4f}'.format(epoch, NUM_EPOCH, tra_loss, tra_acc))
            # deep copy the model
            if tra_acc > best_acc:
                best_acc = tra_acc
                best_model = copy.deepcopy(model)
                torch.save(model, 'epochs/model_{:02}.pth'.format(i))
        print('Best Acc: {:.4f}'.format(best_acc))
        eval(best_model, i)
    acc(ENSEMBLE_SIZE, recall_ids)
