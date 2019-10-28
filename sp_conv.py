import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation='swish', dummy = False):
        super(SpConvBlock, self).__init__()
        assert activation in ['relu', 'swish'] or activation is None
        self.apply_dummy = dummy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.dummy = []
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kh, self.kw = kernel_size

        if isinstance(padding, int):
            self.ph = self.pw = padding
        else:
            assert len(padding) == 2
            self.ph, self.pw = padding
        #self.padding = padding

        if isinstance(stride, int):
            self.dh = self.dw = stride
        else:
            assert len(stride) == 2
            self.dh, self.dw = stride
        #assert stride == 1

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(self.kh, self.kw), \
                                    stride=(self.dh, self.dw), padding=(self.ph, self.pw), bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self._initialize_weights()


    def _initialize_weights(self):
        n = self.conv2d.weight.data.shape[2] * self.conv2d.weight.data.shape[3] * self.conv2d.weight.data.shape[0]
        self.conv2d.weight.data.normal_(0, math.sqrt(2. / n))



    def extract_image_patches(self, x):

        # Pad tensor to get the same output
        x = F.pad(x, (self.pw, self.pw, self.ph, self.ph))

        # get all image windows of size (kh, kw) and stride (dh, dw)
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        # Permute so that channels are next to patch dimension
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        return patches

    def _swish(self, x):
        return x * torch.sigmoid(x)

    def _d_swish(self, x):
        s = torch.sigmoid(x)
        return  s + x * s * (1. - s)

    def _dd_swish(self, x):
        s = torch.sigmoid(x)
        return s*(1.-s) + s + x*s*(1.-s) - (s**2 +2.*x*s**2*(1.-s))


    def _dd_softplus(self, x, beta=3.):
        z = x * beta
        o = F.sigmoid(z)
        return beta * o * (1. - o)


    def forward(self, input):
        r"""
            regular forward
        """
        #print('regular')
        bn_out = self.bn(self.conv2d(input))
        if self.activation is None:
            out = bn_out
        elif self.activation == 'swish':
            out = self._swish(bn_out)
        else:
            out = F.relu(bn_out)
        return out # batch_size * n_out * h * w


    def sp_forward(self, input):
        self.bn.eval()  # fix all bn weights!!!

        conv_out = self.conv2d(input)  # batch size * n_out * h * w
        bn_out = self.bn(conv_out)  # batch size * n_out * h * w
        if self.apply_dummy == False:
            #print('no dummy')
            return F.relu(bn_out)
        # batch normalization
        # y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight  # n_out
        bn_coff = bn_coff.view(1, -1, 1, 1)  # 1, n_out, 1, 1

        if self.activation is None:
            act_sec_ord_grad = torch.ones_like(bn_out)
        elif self.activation == 'swish':
            act_sec_ord_grad = self._dd_swish(bn_out)  # batch_size * n_out * h * w
        else:
            act_sec_ord_grad = self._dd_softplus(bn_out)  # batch_size * n_out * h * w

        bn_coff = bn_coff.permute([0, 2, 3, 1])
        act_sec_ord_grad = act_sec_ord_grad.permute([0, 2, 3, 1])

        # now consider the inner linear terms, that would be the inner product between
        patches = self.extract_image_patches(input)  # [batch_size, h, w, n_in, kh, kw]
        # the second order gradient can be view as a running sum over [h, w] independently
        batch_size, h, w, n_in, kh, kw = patches.size()
        
        patches = patches.view(*patches.size()[:1], -1)
        #print(kh, kw)
        dw = patches.view([-1, self.kh * self.kw * n_in]).contiguous()
        n_out = self.out_channels
        dim = n_in * kh * kw
        self.dummy = []
        for i in range(n_out):
            # dummy variable
            V = Variable(torch.zeros(
                [dim, dim]).cuda(),
                         requires_grad=True)
            self.dummy.append(V)
            #print('appended')
            left1 = act_sec_ord_grad[:, :, :, i:i + 1]
            Bd = bn_coff[:, :, :, i:i + 1]
            left2 = Bd * dw.view([-1, h, w, dim])
            left2 = left2.view([-1, dim])
            tmp = torch.mm(left2, V)
            right = torch.unsqueeze(
                tmp.view([-1, h, w, dim]), 3)
            right = torch.matmul(right, torch.unsqueeze(
                Bd * dw.view([-1, h, w, dim]), -1))

            dummy_term = left1[:, :, :, 0] * right[:, :, :, 0, 0]
            dummy_term = torch.unsqueeze(dummy_term, -1)
            if i == 0:
                Dummy_term = dummy_term
            else:
                Dummy_term = torch.cat([Dummy_term, dummy_term], -1)
        aux = Dummy_term.permute([0, 3, 1, 2])

        if self.activation is None:
            out = bn_out + aux
        elif self.activation == 'swish':
            out = self._swish(bn_out) + aux
        else:
            if self.apply_dummy:
                out = F.relu(bn_out) + aux
            else:
                out =  F.relu(bn_out)
        return out
