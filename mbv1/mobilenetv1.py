'''MobileNetV1 in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Swish, _make_divisible

class MbBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, activation='relu'):
        super(MbBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.nn_act = Swish() if activation == 'swish' else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.nn_act(self.bn1(self.conv1(x)))
        out = self.nn_act(self.bn2(self.conv2(out)))

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, activation='swish'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

        self.nn_act = Swish() if activation == 'swish' else nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.nn_act(self.bn(self.conv(x)))
        return out


defaultcfg = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]


class MobileNetV1(nn.Module):
    def __init__(self, dataset='cifar10', cfg=None, width_mul=None, activation='relu'):
        super(MobileNetV1, self).__init__()

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise NotImplementedError
        self.num_classes = num_classes

        self.activation = activation

        if cfg is None:
            cfg = defaultcfg

        self.cfg = cfg

        self.conv_block = ConvBlock(3, cfg[0], activation=self.activation)
        self.layers = self._make_layers(in_planes=cfg[0])
        self.linear = nn.Linear(cfg[-1], num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for i, x in enumerate(self.cfg[1:]):
            out_planes = x
            if (i+1) in [2, 4, 6, 12]:
                stride = 2
            else:
                stride = 1
            layers.append(MbBlock(in_planes, out_planes, stride, activation=self.activation))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
