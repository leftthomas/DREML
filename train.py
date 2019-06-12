import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

import utils
from model import Model


def processor(sample):
    data, labels, training = sample

    data, labels = data.to(DEVICE), labels.to(DEVICE)

    model.train(training)

    classes = model(data)
    loss = loss_criterion(classes, labels)
    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_loss.reset()
    meter_recall.reset()


def on_forward(state):
    meter_loss.add(state['loss'].item())
    meter_recall.add(state['output'].detach().cpu(), state['sample'][1])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    loss_logger.log(state['epoch'], meter_loss.value()[0], name='train')
    for index, k in enumerate(recall_ids):
        recall_logger.log(state['epoch'], meter_recall.value()[index], name='train_recall_{}'.format(str(k)))
    results['train_loss'].append(meter_loss.value()[0])
    for index, k in enumerate(recall_ids):
        results['train_recall_{}'.format(str(k))].append(meter_recall.value()[index])
    desc = '[Epoch %d] Training Loss: %.4f' % (state['epoch'], meter_loss.value()[0])
    for index, k in enumerate(recall_ids):
        desc += ' Recall@%d: %.2f%%' % (k, meter_recall.value()[index])
    print(desc)

    reset_meters()

    with torch.no_grad():
        engine.test(processor, test_loader)

    loss_logger.log(state['epoch'], meter_loss.value()[0], name='test')
    for index, k in enumerate(recall_ids):
        recall_logger.log(state['epoch'], meter_recall.value()[index], name='test_recall_{}'.format(str(k)))
    results['test_loss'].append(meter_loss.value()[0])
    for index, k in enumerate(recall_ids):
        results['test_recall_{}'.format(str(k))].append(meter_recall.value()[index])
    desc = '[Epoch %d] Testing Loss: %.4f' % (state['epoch'], meter_loss.value()[0])
    for index, k in enumerate(recall_ids):
        desc += ' Recall@%d: %.2f%%' % (k, meter_recall.value()[index])
    print(desc)

    # save model
    torch.save(model.state_dict(), 'epochs/%s_%d.pth' % (DATA_TYPE, state['epoch']))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, state['epoch'] + 1))
    data_frame.to_csv('statistics/{}_results.csv'.format(DATA_TYPE), index_label='epoch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_type', default='cars', type=str, choices=['cars', 'cub', 'sop'], help='dataset type')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

    opt = parser.parse_args()

    DATA_TYPE, RECALLS, BATCH_SIZE, NUM_EPOCH = opt.data_type, opt.recalls, opt.batch_size, opt.num_epochs
    recall_ids = [int(k) for k in RECALLS.split(',')]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {'train_loss': [], 'test_loss': []}
    for k in recall_ids:
        results['train_recall_{}'.format(str(k))], results['test_recall_{}'.format(str(k))] = [], []

    train_loader, test_loader = utils.load_data(data_type=DATA_TYPE, batch_size=BATCH_SIZE)
    model = Model().to(DEVICE)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_recall = utils.RecallMeter(topk=recall_ids)

    loss_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Loss'})
    recall_logger = VisdomPlotLogger('line', env=DATA_TYPE, opts={'title': 'Recall'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCH, optimizer=optimizer)
