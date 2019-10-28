from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import sys
from numpy import linalg as LA
from dataloader import  get_data_loader
from compute_flops import print_model_param_nums, print_model_param_flops
from sp_mbnet import SpMbBlock, SpConvBlock
import json

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)', required=True)
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to the pruned/split model to be fine tuned', required=True)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--sp', action='store_true', default=True,
                    help='splitting settings')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='retrain, otherwise, finetune')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--layer', type=int, default=-1,
                    help='layer to split')
parser.add_argument('--warm', type=int, default=0,
                    help='warm up the training')
parser.add_argument('--rd', type=int, default=0,
                    help='split round')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.sp:
    from sp_mbnet import sp_mbnet as mbnet
else:
    from mobilenetv1 import MobileNetV1 as mbnet

assert args.load
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.save = os.path.dirname(args.load)
if args.retrain:
    args.save = os.path.join(args.save, 'retrain')
else:
    args.save = os.path.join(args.save, 'finetune')

if not os.path.exists(args.save):
    os.makedirs(args.save)

checkpoint = torch.load(args.load)

if args.sp and args.layer == -1:
    model_save_path = os.path.join(args.save,'{}_{}.pth.tar'.format( args.dataset, str(int(args.rd) + 1)))

if args.layer == -1:
    logging_file_path = model_save_path.replace(".pth.tar", ".log")
else:
    logging_file_path = str(args.layer) + '.log'

print('====================ROUND========================', args.rd)
#########################################################
# create file handler which logs even debug messages
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(logging_file_path)

formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################



from dataloader import get_data_loader
train_loader, test_loader = \
        get_data_loader(dataset = args.dataset, train_batch_size = args.batch_size, test_batch_size = args.test_batch_size, use_cuda=args.cuda)

model = mbnet(cfg=checkpoint['cfg'], dataset=args.dataset, dummy_layer = args.layer)
# load weights, otherwise, only the arch is used
print(model.cfg)
if not os.path.exists('config'):
    os.makedirs('config')
if args.layer != -1:
    pickle.dump(model.cfg, open('config/{}_{}.pkl'.format(args.dataset, str(args.rd)), 'wb'))
else:
    pickle.dump(model.cfg, open('config/{}_{}.pkl'.format(args.dataset, str(args.rd + 1)), 'wb'))

if not args.retrain:
    model.load_state_dict(checkpoint['state_dict'])

if args.cuda: model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.warm == 1:
            if epoch == 0:
                for params_group in optimizer.param_groups:
                    params_group['lr'] = args.lr * (batch_idx / 23)
            if epoch == 1:
                for params_group in optimizer.param_groups:
                    params_group['lr'] = args.lr

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.data
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def compute_A(epoch):
    model.eval()
    avg_loss = 0.
    train_acc = 0.
    count = 0
    A = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model.sp_forward(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.data
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
        dummy_parameters = []
        layer = args.layer
        dummy_parameters += model.layers[layer].dummy
        loss.backward()
        a = [item.grad.data.cpu().numpy() for item in model.layers[layer].dummy]
        a = np.array(a)
        if count == 0:
            A = a
        else:
            A += a
        count += 1
    A = np.array(A)
    A = A / count
    print(A.shape)
    rd = args.rd
    calculate_eigen(A, args.layer, rd)


def calculate_eigen(A, layer, rd = 0):

    A = (A + np.transpose(A, [0, 2, 1])) / 2
    w, v = LA.eig(A)
    w_min = np.min(w, axis=1)
    V = []
    for a in range(w_min.shape[0]):
        amina = np.argmin(w[a])
        V.append(v[a, :, amina])
    V = np.array(V)
    if not os.path.exists('eigen'):
        os.makedirs('eigen')
    pickle.dump(w_min, open('eigen/{}_A_{}_{}_.pkl'.format(args.dataset, str(layer), str(rd)), 'wb'))
    pickle.dump(V, open('eigen/{}_V_{}_{}_.pkl'.format(args.dataset, str(layer), str(rd)), 'wb'))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


prec1 = test()
best_prec1 = 0.
for epoch in range(args.epochs):

    if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    if args.layer != -1:
        compute_A(epoch)
        break
    train(epoch)
    prec1 = test()
    best_prec1 = max(prec1, best_prec1)

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'cfg': model.cfg,
        'optimizer': optimizer.state_dict(),
        'acc': prec1,
    }, model_save_path)


print("Best accuracy: "+str(best_prec1))
if not os.path.exists('result'):
    os.makedirs('result')
pickle.dump([prec1, best_prec1], open('result/{}_{}_result.pkl'.format(args.dataset, str(args.rd)),'wb'))
