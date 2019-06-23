import argparse

import pandas as pd
import torch
import torch.optim as optim
import torchnet as tnt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

import utils
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='cars', type=str, choices=['cars', 'cub', 'sop'], help='dataset name')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCH = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    recall_ids = [int(k) for k in RECALLS.split(',')]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {'train_loss': []}
    for k in recall_ids:
        results['train_recall_{}'.format(k)], results['test_recall_{}'.format(k)] = [], []

    train_set = utils.RetrievalDataset(DATA_NAME, data_type='train')
    val_set = utils.RetrievalDataset(DATA_NAME, data_type='val')
    test_set = utils.RetrievalDataset(DATA_NAME, data_type='test')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)

    # load data to memory
    print('load data to memory, it may take a while')
    val_database, val_labels, test_database, test_labels = [], [], [], []
    for img, label, index in val_loader:
        val_database.append(img)
        val_labels.append(label)
    val_labels = torch.cat(val_labels)
    for img, label, index in test_loader:
        test_database.append(img)
        test_labels.append(label)
    test_labels = torch.cat(test_labels)

    model = Model(len(train_set)).to(DEVICE)
    loss_criterion = CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    meter_loss = tnt.meter.AverageValueMeter()
    meter_recall = utils.RecallMeter(topk=recall_ids)
    loss_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Loss'})
    recall_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Recall'})

    for epoch in range(1, NUM_EPOCH + 1):
        # train loop
        model.train()
        train_progress, num_data = tqdm(train_loader), 0
        for img, label, index in train_progress:
            num_data += img.size(0)
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            feature, out = model(img)
            loss = loss_criterion(out, label)
            loss.backward()
            optimizer.step()
            meter_loss.add(loss.item())
            train_progress.set_description('Train Epoch: {}---{}/{} Loss: {:.2f}'.format(
                epoch, num_data, len(train_set), meter_loss.value()[0]))
        loss_logger.log(epoch, meter_loss.value()[0], name='train')
        results['train_loss'].append(meter_loss.value()[0])
        meter_loss.reset()

        # test loop
        with torch.no_grad():
            model.eval()
            val_features = []
            for data in val_database:
                data = data.to(DEVICE)
                val_feature = model.features(data).view(data.size(0), -1)
                val_features.append(val_feature.detach().cpu())
            val_features = torch.cat(val_features)
            # compute recall for train data
            val_progress, num_data = tqdm(val_loader), 0
            for img, label, index in val_progress:
                num_data += img.size(0)
                img = img.to(DEVICE)
                out = model.features(img).view(img.size(0), -1).detach().cpu()
                meter_recall.add(out, index, label, val_features, val_labels)
                desc = 'Val Epoch: {}---{}/{}'.format(epoch, num_data, len(val_set))
                for i, k in enumerate(recall_ids):
                    desc += ' Recall@{}: {:.2f}%'.format(k, meter_recall.value()[i])
                val_progress.set_description(desc)
            for i, k in enumerate(recall_ids):
                recall_logger.log(epoch, meter_recall.value()[i], name='train_recall_{}'.format(k))
                results['train_recall_{}'.format(k)].append(meter_recall.value()[i])
            meter_recall.reset()

            test_features = []
            for data in test_database:
                data = data.to(DEVICE)
                test_feature = model.features(data).view(data.size(0), -1)
                test_features.append(test_feature.detach().cpu())
            test_features = torch.cat(test_features)
            # compute recall for test data
            test_progress, num_data = tqdm(test_loader), 0
            for img, label, index in test_progress:
                num_data += img.size(0)
                img = img.to(DEVICE)
                out = model.features(img).view(img.size(0), -1).detach().cpu()
                meter_recall.add(out, index, label, test_features, test_labels)
                desc = 'Test Epoch: {}---{}/{}'.format(epoch, num_data, len(test_set))
                for i, k in enumerate(recall_ids):
                    desc += ' Recall@{}: {:.2f}%'.format(k, meter_recall.value()[i])
                test_progress.set_description(desc)
            for i, k in enumerate(recall_ids):
                recall_logger.log(epoch, meter_recall.value()[i], name='test_recall_{}'.format(k))
                results['test_recall_{}'.format(k)].append(meter_recall.value()[i])
            meter_recall.reset()

        # save model
        torch.save(model.state_dict(), 'epochs/{}_{}.pth'.format(DATA_NAME, epoch))
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_NAME), index_label='epoch')
