import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm

import utils
from model import Network


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
    meter_accuracy.reset()
    meter_loss.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][1])
    meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    loss_logger.log(state['epoch'], meter_loss.value()[0], name='train')
    accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='train')
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy'].append(meter_accuracy.value()[0])
    print('[Epoch %d] Training Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    reset_meters()

    with torch.no_grad():
        engine.test(processor, test_loader)

    loss_logger.log(state['epoch'], meter_loss.value()[0], name='test')
    accuracy_logger.log(state['epoch'], meter_accuracy.value()[0], name='test')
    results['test_loss'].append(meter_loss.value()[0])
    results['test_accuracy'].append(meter_accuracy.value()[0])
    print('[Epoch %d] Testing Loss: %.4f Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    # save model
    torch.save(model.state_dict(), 'epochs/%d.pth' % (state['epoch']))
    # save statistics
    data_frame = pd.DataFrame(
        data={'train_loss': results['train_loss'], 'train_accuracy': results['train_accuracy'],
              'test_loss': results['test_loss'], 'test_accuracy': results['test_accuracy']},
        index=range(1, state['epoch'] + 1))
    data_frame.to_csv('statistics/results.csv', index_label='epoch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Human Matting Model')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

    opt = parser.parse_args()

    BATCH_SIZE = opt.batch_size
    NUM_EPOCH = opt.num_epochs
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}

    train_loader, test_loader = utils.load_data(batch_size=BATCH_SIZE)
    model = Network().to(DEVICE)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters())
    print("# parameters:", sum(param.numel() for param in model.parameters()))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)

    loss_logger = VisdomPlotLogger('line', opts={'title': 'Loss'})
    accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Accuracy'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCH, optimizer=optimizer)
