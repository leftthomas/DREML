import csv
import copy
import argparse

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchnet.logger import VisdomPlotLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader

from model import Model
from utils import ImageReader, create_id, get_transform, load_data, recall


# screen python train.py --num_epochs=12 --data_name='car' --classifier_type='linear' --ensemble_size=48


def write_csv(data, is_first_time):
    if is_first_time:
        with open('results/{}/result.csv'.format(DATA_NAME), 'w', newline='') as csvfile:
            fieldnames = ['Model', 'Epoch', 'Loss', 'Recall_1', 'Recall_2', 'Recall_4', 'Recall_8']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    else:
        with open('results/{}/result.csv'.format(DATA_NAME), 'a+', newline='') as csvfile:
            fieldnames = ['flag', 'epoch', 'loss', 'Recall_1', 'Recall_2', 'Recall_4', 'Recall_8']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(data)


def train(net, data_dict, optim, model_id, epoch_id):
    net.train()
    data_set = ImageReader(data_dict, get_transform(DATA_NAME, 'train'))
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    l_data, t_data, n_data = 0.0, 0, 0
    for inputs, labels in data_loader:
        optim.zero_grad()
        out = net(inputs.to(DEVICE))
        loss = criterion(out, labels.to(DEVICE))
        print('loss:{:.4f}'.format(loss.item()), end='\r')
        loss.backward()
        optim.step()
        _, pred = torch.max(out, 1)
        l_data += loss.item()
        t_data += torch.sum(pred.cpu() == labels).item()
        n_data += len(labels)

    data = {'flag': model_id, 'epoch': epoch_id, 'loss': format(l_data / n_data, '.6f'),
            'Recall_1': '', 'Recall_2': '', 'Recall_4': '', 'Recall_8': ''}
    write_csv(data, False)
    loss_logger.log(epoch_id, l_data / n_data, name='train Model {}'.format(model_id))
    return l_data / n_data, t_data / n_data


def eval(net, data_dict, ensemble_num, recalls):
    net.eval()
    data_set = ImageReader(data_dict, get_transform(DATA_NAME, 'test'))
    data_loader = DataLoader(data_set, BATCH_SIZE, shuffle=False, num_workers=8)

    features = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            out = net(inputs.to(DEVICE))
            out = F.normalize(out)
            features.append(out.cpu())
    features = torch.cat(features, 0)
    torch.save(features,
               'results/{}/{}_{}_test_features_{:03}.pth'.format(DATA_NAME, DATA_NAME, CLASSIFIER_TYPE, ensemble_num))
    # load feature vectors
    features = [torch.load('results/{}/{}_{}_test_features_{:03}.pth'.format(DATA_NAME, DATA_NAME, CLASSIFIER_TYPE, d))
                for d in range(1, ensemble_num + 1)]
    features = torch.cat(features, 1)
    acc_list = recall(features, data_set.labels, rank=recalls)
    desc = ''
    for index, recall_id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(recall_id, acc_list[index] * 100)
        recall_logger.log(ensemble_num, acc_list[index], name='Recall_{}'.format(recall_id))
    data = {'flag': ensemble_num, 'epoch': '', 'loss': '',
            'Recall_1': format(acc_list[0] * 100, '.6f'),
            'Recall_2': format(acc_list[1] * 100, '.6f'),
            'Recall_4': format(acc_list[2] * 100, '.6f'),
            'Recall_8': format(acc_list[3] * 100, '.6f')}
    write_csv(data, False)
    print(desc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop'], help='dataset name')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=12, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=12, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='meta class size')
    parser.add_argument('--classifier_type', default='capsule', type=str, choices=['capsule', 'linear'],
                        help='classifier type')

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCHS = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    ENSEMBLE_SIZE, META_CLASS_SIZE, CLASSIFIER_TYPE = opt.ensemble_size, opt.meta_class_size, opt.classifier_type
    recall_ids = [int(k) for k in RECALLS.split(',')]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dicts = torch.load('data/{}/data_dicts.pth'.format(DATA_NAME))
    train_data, test_data = data_dicts['train'], data_dicts['test']
    # sort classes and fix the class order
    all_class = sorted(train_data)
    idx_to_class = {i: all_class[i] for i in range(len(all_class))}

    # set Visdom
    loss_logger = VisdomPlotLogger('line', env='DCN_lsy_{}'.format(DATA_NAME), opts={'title': 'Loss'})
    recall_logger = VisdomPlotLogger('line', env='DCN_lsy_{}'.format(DATA_NAME), opts={'title': 'Recall'})

    # create csv
    write_csv(None, True)

    for i in range(1, ENSEMBLE_SIZE + 1):
        print('Training ensemble #{}'.format(i))
        meta_id = create_id(META_CLASS_SIZE, len(data_dicts['train']))
        meta_data_dict = load_data(meta_id, idx_to_class, train_data)
        model = Model(META_CLASS_SIZE, CLASSIFIER_TYPE).to(DEVICE)

        # optim_configs = [{'params': model.features.parameters(), 'lr': 1e-4 * 10},
        #                  {'params': model.fc.parameters(), 'lr': 1e-4}]
        optimizer = Adam(model.parameters(), lr=1e-3)
        lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)
        criterion = CrossEntropyLoss()

        best_acc, best_model = 0, None
        for epoch in range(1, NUM_EPOCHS + 1):
            lr_scheduler.step(epoch)
            train_loss, train_acc = train(model, meta_data_dict, optimizer, i, epoch)
            print('Epoch {}/{} - Loss:{:.4f} - Acc:{:.4f}'.format(epoch, NUM_EPOCHS, train_loss, train_acc))
            # deep copy the model
            if train_acc > best_acc:
                best_acc = train_acc
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(),
                           'epochs/{}/{}_{}_model_{:03}.pth'.format(DATA_NAME, DATA_NAME, CLASSIFIER_TYPE, i))
        eval(best_model, test_data, i, recall_ids)

