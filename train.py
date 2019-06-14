import argparse

import pandas as pd
import torch
import torch.optim as optim
import torchnet as tnt
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
        results['train_recall_{}'.format(str(k))], results['test_recall_{}'.format(str(k))] = [], []

    train_set = utils.RetrievalDataset(DATA_NAME, data_type='train')
    val_set = utils.RetrievalDataset(DATA_NAME, data_type='val')
    test_set = utils.RetrievalDataset(DATA_NAME, data_type='test')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # load data to memory
    print('load data to memory, it may take a while')
    val_database, test_database = [], []
    for img, label, index in DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=16, shuffle=False):
        val_database.append(img)
    for img, label, index in DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=16, shuffle=False):
        test_database.append(img)

    model = Model().to(DEVICE)
    loss_criterion = utils.DiverseLoss()
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
        for img, positives, negatives in train_progress:
            num_data += img.size(0)
            img, positives, negatives = img.to(DEVICE), positives.to(DEVICE), negatives.to(DEVICE)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_criterion(out, positives, negatives, model)
            loss.backward()
            optimizer.step()
            meter_loss.add(loss.item())
            train_progress.set_description('Train Epoch: {}---{}/{} Loss: {:.2f}'.format(
                epoch, num_data, len(train_set), meter_loss.value()[0]))
        loss_logger.log(epoch, meter_loss.value()[0], name='train')
        results['train_loss'].append(meter_loss.value()[0])
        print('Train Epoch: {} Loss: {:.2f}'.format(epoch, meter_loss.value()[0]))
        meter_loss.reset()

        # test loop
        with torch.no_grad():
            model.eval()
            # compute recall for train data
            val_progress, num_data = tqdm(val_loader), 0
            for img, label, index in val_progress:
                num_data += img.size(0)
                img = img.to(DEVICE)
                out = model(img)
                val_features = []
                for data in val_database:
                    val_features.append(model(data))
                val_features = torch.cat(val_features)
                meter_recall.add(out.detach().cpu(), index, list(label), val_features)
                desc = 'Val Epoch: {}---{}/{}'.format(epoch, num_data, len(val_set))
                for i, k in enumerate(recall_ids):
                    desc += ' Recall@%d: %.2f%%' % (k, meter_recall.value()[i])
                val_progress.set_description(desc)
            desc = 'Val Epoch: {}'.format(epoch)
            for i, k in enumerate(recall_ids):
                recall_logger.log(epoch, meter_recall.value()[i], name='train_recall_{}'.format(str(k)))
                results['train_recall_{}'.format(str(k))].append(meter_recall.value()[i])
                desc += ' Recall@%d: %.2f%%' % (k, meter_recall.value()[i])
            print(desc)
            meter_recall.reset()

            # compute recall for test data
            test_progress, num_data = tqdm(test_loader), 0
            for img, label, index in test_progress:
                num_data += img.size(0)
                img = img.to(DEVICE)
                out = model(img)
                test_features = []
                for data in test_database:
                    test_features.append(model(data))
                test_features = torch.cat(test_features)
                meter_recall.add(out.detach().cpu(), index, list(label), test_features)
                desc = 'Test Epoch: {}---{}/{}'.format(epoch, num_data, len(test_set))
                for i, k in enumerate(recall_ids):
                    desc += ' Recall@%d: %.2f%%' % (k, meter_recall.value()[i])
                test_progress.set_description(desc)
            desc = 'Test Epoch: {}'.format(epoch)
            for i, k in enumerate(recall_ids):
                recall_logger.log(epoch, meter_recall.value()[i], name='test_recall_{}'.format(str(k)))
                results['test_recall_{}'.format(str(k))].append(meter_recall.value()[i])
                desc += ' Recall@%d: %.2f%%' % (k, meter_recall.value()[i])
            print(desc)
            meter_recall.reset()

        # save model
        torch.save(model.state_dict(), 'epochs/%s_%d.pth' % (DATA_NAME, epoch))
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_NAME), index_label='epoch')
