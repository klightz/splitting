'''MobileNetV1 in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Swish
from sp_conv import SpConvBlock

__all__ = ['sp_mbnet']


splitcfg = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]


class SpMbBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, activation='swish', dummy = False):
        super(SpMbBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        #print(dummy)
        self.sp_conv = SpConvBlock(in_planes, out_planes, kernel_size=1, stride=1, padding=0, activation=activation, dummy = dummy)
        self.dummy = []
        self.nn_act = Swish() if activation == 'swish' else nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.nn_act(self.bn1(self.conv1(x)))
        out = self.sp_conv(out)
        return out

    def sp_forward(self, x):
        out = self.nn_act(self.bn1(self.conv1(x)))
        out = self.sp_conv.sp_forward(out)
        self.dummy = self.sp_conv.dummy
        return out



class sp_mbnet(nn.Module):

    def __init__(self, dataset='cifar10', cfg=None, activation='relu', dummy_layer = -1 ):
        super(sp_mbnet, self).__init__()
        self.dummy_layer = dummy_layer

        if dataset == 'cifar10' or dataset == 'mnist' or dataset == 'fashion_mnist':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise NotImplementedError
        self.num_classes = num_classes

        self.activation = activation
        if cfg is None:
            cfg = splitcfg

        self.cfg = cfg

        self.layers = self._make_layers()
        self.linear = nn.Linear(cfg[-1], num_classes)


    def _make_layers(self,):
        layers = []

        in_planes = self.cfg[0]
        if self.dummy_layer == 0:
            layers += [SpConvBlock(3, in_planes, kernel_size=3, stride=1, padding=1, activation=self.activation, dummy = True)]
            #print('ZERO')
        else:
            layers += [
                SpConvBlock(3, in_planes, kernel_size=3, stride=1, padding=1, activation=self.activation, dummy=False)]

        for i, x in enumerate(self.cfg[1:]):
            if (i + 1) == self.dummy_layer:
                add_dummy = True
                print(i + 1)
                print(len(layers))
            else:
                add_dummy = False
            if (i+1) in [2, 4, 6, 12]:
                st = 2
            else:
                st = 1
            out_planes = x if isinstance(x, int) else x[0]

            layers.append(SpMbBlock(in_planes, out_planes, stride= st, activation=self.activation, dummy = add_dummy))
            in_planes = out_planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def sp_forward(self, x):
        out = x
        for l in self.layers:
            out = l.sp_forward(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
