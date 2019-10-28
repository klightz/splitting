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

import sys
from dataloader import  get_data_loader
from sp_mbnet import sp_mbnet as mbnet
from sp_mbnet import SpMbBlock, SpConvBlock
from compute_flops import print_model_param_nums, print_model_param_flops

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--load', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)', required=True)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--split-index', default="1", type=str,
                    help='#number of split', required=True)
parser.add_argument('--energy', action='store_true', default=False,
                    help='energy aware splitting')
parser.add_argument('--params', action='store_true', default=False,
                    help='paramter aware splitting')
parser.add_argument('--save', default='split/saved_models', type=str,
                    help='energy aware splitting')
parser.add_argument('--grow', type=float, default=0.3,
                    help='globally split grow rate (default: 0.2)')
parser.add_argument('--exp-name', type=str, default=None,
                    help='exp name', required=True)
parser.add_argument('--start-from-retrain', action='store_true', default=False,
                    help='retrain before splitting')
parser.add_argument('--rd', type=int,default = 0,
                    help='split round')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = \
        get_data_loader(dataset = args.dataset, train_batch_size = args.batch_size, test_batch_size = args.test_batch_size, use_cuda=args.cuda)

args.save = os.path.join(args.save, args.exp_name)


logging_file_path = '{}_split_{}.log'.format(args.dataset, args.split_index)
model_save_path = '{}_{}.pth.tar'.format(args.dataset, str(args.rd))

if not os.path.exists(args.save):
    os.makedirs(args.save)

#########################################################
# create file handler which logs even debug messages
import logging
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(args.save, logging_file_path))

formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)
#########################################################

assert args.load
assert os.path.isfile(args.load)
log.info("=> loading checkpoint '{}'".format(args.load))
checkpoint = torch.load(args.load)
model = mbnet(dataset=args.dataset, cfg=checkpoint['cfg']).to(device)

from collections import OrderedDict
new_state_dict = OrderedDict()

model.load_state_dict(checkpoint['state_dict'])
log.info("=> loaded checkpoint '{}' " .format(args.load))
del checkpoint

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #output = model.sp_forward(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()

    test_loss /= len(test_loader.dataset)
    log.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / float(len(test_loader.dataset))))
    return correct / float(len(test_loader.dataset))


log.info('acc before splitting')
test(model)
import pickle
num = str(args.rd)
base = model.cfg
cfg_mask = pickle.load(open('eigen/{}_min.pkl'.format(num), 'rb'))
min_eig_vecs = pickle.load(open('eigen/{}_minv.pkl'.format(num), 'rb'))
cfg = pickle.load(open('config/delta_{}_{}.pkl'.format(args.dataset, str(int(num) + 1)), 'rb'))
cfg = cfg.tolist()

print(base)
for i in range(len(base)):
    cfg[i] +=  base[i]
print(cfg)

##################################
##### copy weights and split #####
##################################
newmodel = mbnet(dataset=args.dataset, cfg=cfg)
newmodel.to(device)

layer_id_in_cfg = 0
start_mask = np.array([])
end_mask = cfg_mask[layer_id_in_cfg]
for k, (m0, m1) in enumerate(zip(model.modules(), newmodel.modules())):
    if k == 2: assert isinstance(m0, SpConvBlock)
    if (k==2 and isinstance(m0, SpConvBlock)) or isinstance(m0, SpMbBlock):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        idx0 = np.squeeze(np.asarray(start_mask))
        idx1 = np.squeeze(np.asarray(end_mask))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if isinstance(m0, SpMbBlock):
            # conv1, depth wise conv
            conv_weight = m0.conv1.weight.data.clone()
            m1.conv1.weight.data = torch.cat((conv_weight, conv_weight[idx0.tolist(), :, :, :].clone()), 0)

            # bn1
            m1.bn1.weight.data = torch.cat((m0.bn1.weight.data.clone(), m0.bn1.weight.data[idx0.tolist()].clone()))
            m1.bn1.bias.data = torch.cat((m0.bn1.bias.data.clone(), m0.bn1.bias.data[idx0.tolist()].clone()))
            m1.bn1.running_mean =torch.cat((m0.bn1.running_mean.clone(), m0.bn1.running_mean[idx0.tolist()].clone()))
            m1.bn1.running_var = torch.cat((m0.bn1.running_var.clone(), m0.bn1.running_var[idx0.tolist()].clone()))

        if isinstance(m0, SpMbBlock):
            m0_conv = m0.sp_conv.conv2d
            m1_conv = m1.sp_conv.conv2d
            m0_bn = m0.sp_conv.bn
            m1_bn = m1.sp_conv.bn
        else:
            m0_conv = m0.conv2d
            m1_conv = m1.conv2d
            m0_bn = m0.bn
            m1_bn = m1.bn

        # copy conv weights
        conv_weight = m0_conv.weight.data.clone()
        if idx0.size != 0:
        	conv_weight[:, idx0.tolist(), :, :] /= 2.
        w1 = torch.cat((conv_weight, conv_weight[:, idx0.tolist(), :, :]), 1)
        w1 = torch.cat((w1.clone(), w1[idx1.tolist(), :, :, :].clone()), 0)
        eig_v = min_eig_vecs[layer_id_in_cfg].astype(float)
        eig_v = torch.from_numpy(eig_v).float().cuda()
        if idx0.size != 0:
               eig_v[:, idx0.tolist(), :, :] /= 2.
        eig_v = torch.cat((eig_v, eig_v[:, idx0.tolist(), :, :]), 1)
        w1[idx1.tolist(), :, :, :] += 1e-2 * eig_v[idx1.tolist(), :, :, :]
        w1[conv_weight.size(0):, :, :, :] -= 1e-2 * eig_v[idx1.tolist(), :, :, :]

        m1_conv.weight.data = w1.clone()

        # copy bn weights
        bn_weight = m0_bn.weight.data.clone()
        m1_bn.weight.data = torch.cat((bn_weight.clone(), bn_weight[idx1.tolist()].clone()))

        bn_bias = m0_bn.bias.data.clone()
        m1_bn.bias.data = torch.cat((bn_bias.clone(), bn_bias[idx1.tolist()].clone()))

        bn_running_mean = m0_bn.running_mean.clone()
        m1_bn.running_mean = torch.cat((bn_running_mean.clone(), bn_running_mean[idx1.tolist()].clone()))

        bn_running_var = m0_bn.running_var.clone()
        m1_bn.running_var = torch.cat((bn_running_var.clone(), bn_running_var[idx1.tolist()].clone()))

        layer_id_in_cfg += 1
        start_mask  = end_mask
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.asarray(start_mask))
        #if idx0.size == 0: continue
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        fc_weight = m0.weight.data.clone()
        fc_weight[:, idx0.tolist()] /= 2.
        fc_bias = m0.bias.data.clone()
        m1.weight.data = torch.cat((fc_weight, fc_weight[:, idx0.tolist()]), 1)
        m1.bias.data = m0.bias.data.clone()

log.info(model.cfg)
log.info(newmodel.cfg)
print('acc after splitting')
test(newmodel)
test(model)
torch.save({
        'cfg': newmodel.cfg,
        'split_index': args.split_index,
        'state_dict': newmodel.state_dict(),
        'args': args,
    }, os.path.join(args.save, model_save_path)
)
print(os.path.join(args.save, model_save_path))
