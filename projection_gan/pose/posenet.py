#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Based on Yasunori Kudo work

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class MLP(nn.Module):

    def __init__(self, n_in=34, n_unit=1024, mode='generator',
                 use_bn=False, activate_func=F.leaky_relu):
        if n_in % 2 != 0:
            raise ValueError("'n_in' must be divisible by 2.")
        if not mode in ['generator', 'discriminator']:
            raise ValueError("only 'generator' and 'discriminator' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        super(MLP, self).__init__()
        n_out = n_in // 2 if mode == 'generator' else 1
        print('MODEL: {}, N_OUT: {}, N_UNIT: {}'.format(mode, n_out, n_unit))
        self.mode = mode
        self.use_bn = use_bn
        self.activate_func = activate_func
        #w = chainer.initializers.Normal(0.02)
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features=n_in, out_features=n_unit)
        self.l2 = nn.Linear(in_features=n_unit, out_features=n_unit)
        self.l3 = nn.Linear(in_features=n_unit, out_features=n_unit)
        self.l4 = nn.Linear(in_features=n_unit, out_features=n_out)

        # https://qiita.com/maskot1977/items/cee6aafb59342af8630c
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(n_unit)
            self.bn2 = nn.BatchNorm1d(n_unit)
            self.bn3 = nn.BatchNorm1d(n_unit)

    def forward(self, x):
        #print('>>> x:{}'.format(x.shape))
        if self.use_bn:
            h1 = self.activate_func(self.bn1(self.l1(x)))
            h2 = self.activate_func(self.bn2(self.l2(h1)))
            h3 = self.activate_func(self.bn3(self.l3(h2)) + h1)
        else:
            h1 = self.activate_func(self.l1(x))
            h2 = self.activate_func(self.l2(h1))
            h3 = self.activate_func(self.l3(h2) + h1)
        return self.l4(h3)
