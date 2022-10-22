#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Yasunori Kudo

from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class H36M_Updater():

    def __init__(self, gan_accuracy_cap, use_heuristic_loss,
                 heuristic_loss_weight, mode, *args, **kwargs):
        if not mode in ['supervised', 'unsupervised']:
            raise ValueError("only 'supervised' and 'unsupervised' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        self.gan_accuracy_cap = gan_accuracy_cap
        self.use_heuristic_loss = use_heuristic_loss
        self.heuristic_loss_weight = heuristic_loss_weight
        self.mode = mode
        super(H36M_Updater, self).__init__(*args, **kwargs)

    @staticmethod
    def calculate_rotation(xy_real, z_pred):
        #xy_split = torch.split(xy_real, split_size_or_sections=xy_real.data.shape[1]) # axis
        #z_split = torch.split(z_pred, split_size_or_sections=z_pred.data.shape[1]) # axis
        xy_split = torch.chunk(xy_real, chunks=xy_real.shape[1], dim=1) # axis
        z_split = torch.chunk(z_pred, chunks=z_pred.shape[0], dim=0) # axis
        #print('xy_real.data.shape[1]:{}'.format(xy_real.data.shape[1]))
        #print('xy_split:{}'.format(xy_real.shape))
        #print('z_split:{}'.format(z_pred.shape))
        #print('xy_split:{}'.format(len(xy_split)))
        #print('z_split:{}'.format(len(z_split)))
        # if the size mismatch, return sine value of 0 (sin(0)=1)
        if len(z_split) < 16:
            print('[e] Code crash here:calculate_rotation')
            exit(0)  # <---- ERROR
        # Vector v0 (neck -> nose) on zx-plain. v0=(a0, b0).
        a0 = z_split[9] - z_split[8]
        b0 = xy_split[9 * 2] - xy_split[8 * 2]
        n0 = torch.sqrt(a0 * a0 + b0 * b0)
        # Vector v1 (right shoulder -> left shoulder) on zx-plain. v1=(a1, b1).
        a1 = z_split[14] - z_split[11]
        b1 = xy_split[14 * 2] - xy_split[11 * 2]
        n1 = torch.sqrt(a1 * a1 + b1 * b1)      
        # Return sine value of the angle between v0 and v1.
        return (a0 * b1 - a1 * b0) / (n0 * n1)

    @staticmethod
    def calculate_heuristic_loss(xy_real, z_pred):
        return torch.mean(F.relu(
            -H36M_Updater.calculate_rotation(xy_real, z_pred)))

    def update_core(self, batchsize, xy_proj, xyz, scale, gen, dis, gen_optimizer, dis_optimizer, device, show):
        # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/2
        # https://github.com/NVlabs/FUNIT/issues/23
        #gen_optimizer = self.get_optimizer('gen')
        #dis_optimizer = self.get_optimizer('dis')
        #gen, dis = gen_optimizer.target, dis_optimizer.target
        if xy_proj.shape[0] < 16:
            return
        #print('xy_proj:{}'.format(xy_proj.shape))
        #batchsize = len(batch)
        #xy_proj, xyz, scale = self.converter(batch, self.device)
        #xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
        torch.autograd.set_detect_anomaly(True)
        xy_real = torch.Tensor(xy_proj).to(device)
        z_pred = gen(xy_real)

        xyz = xyz.to(device)
        xy_real = xy_real.to(device)
        z_pred = z_pred.to(device)

        #print('xyz:{}'.format(type(xyz)))
        #print('xy_real:{}'.format(type(xy_real)))
        #print('z_pred:{}'.format(type(z_pred)))

        #print('z_pred:{} exp:{}'.format(z_pred, xyz[:, 2::3]))

        z_mse = F.mse_loss(z_pred, xyz[:, 2::3])

        if self.mode == 'supervised':
            gen.zero_grad()
            z_mse.backward()
            gen_optimizer.step()
            #chainer.report({'z_mse': z_mse}, gen)

        elif self.mode == 'unsupervised':
            # Random rotation.
            theta = torch.FloatTensor(1, batchsize).uniform_(0, 2 * np.pi).to(device)
            #print('theta:{}'.format(theta.shape))
            cos_theta = torch.cos(theta).view(batchsize,-1)
            sin_theta = torch.sin(theta).view(batchsize,-1)
            #print('cos_theta:{}'.format(cos_theta.shape))
            #print('sin_theta:{}'.format(sin_theta.shape))

            # 2D Projection.
            x = xy_real[:, 0::2]
            y = xy_real[:, 1::2]
            #print('x:{}'.format(x.shape))
            #print('z_pred:{}'.format(z_pred.shape))
            new_x = x * cos_theta + z_pred * sin_theta
            xy_fake = torch.cat((new_x[:, :, None], y[:, :, None]), axis=2)
            xy_fake = torch.reshape(xy_fake, (batchsize, -1))

            y_real = dis(xy_real)
            y_fake = dis(xy_fake)

            #print('xy_real:{}'.format(xy_real))
            #print('xy_fake:{}'.format(xy_fake))

            #print('y_real:{}'.format(y_real))
            #print('y_fake:{}'.format(y_fake))

            acc_dis_fake = torch.sum(torch.eq(
                torch.round(y_fake).view(-1), torch.zeros(y_fake.data.shape, dtype=torch.int).to(device).view(-1)))
            acc_dis_real = torch.sum(torch.eq(
                torch.round(y_real).view(-1), torch.ones(y_real.data.shape, dtype=torch.int).to(device).view(-1)))
            #print('acc_dis_fake:{}'.format(acc_dis_fake))
            #print('acc_dis_real:{}'.format(acc_dis_real))
            acc_dis = (acc_dis_fake + acc_dis_real) / (y_real.view(-1).shape[0] + y_fake.view(-1).shape[0])
            #print('acc_dis:{}'.format(acc_dis))

            loss_gen = torch.sum(F.softplus(-y_fake.clone())) / batchsize
            if self.use_heuristic_loss:
                loss_heuristic = self.calculate_heuristic_loss(
                    xy_real=xy_real, z_pred=z_pred)
                loss_gen += loss_heuristic * self.heuristic_loss_weight
                #chainer.report({'loss_heuristic': loss_heuristic}, gen)
            gen_optimizer.zero_grad()
            if acc_dis.data >= (1 - self.gan_accuracy_cap):
                loss_gen.backward(retain_graph=True)
            #xy_fake.unchain_backward() # no counterpart

            dis_optimizer.zero_grad()
            #loss_dis = torch.add(torch.sum(F.softplus(-y_real)) / batchsize, torch.sum(F.softplus(y_fake)) / batchsize)
            loss_dis = torch.sum(F.softplus(-y_real)) / batchsize
            loss_dis += torch.sum(F.softplus(y_fake)) / batchsize# <- problem y_fake?
            if acc_dis.data <= self.gan_accuracy_cap:
                loss_dis.backward()

            #print('y:{} {}'.format(y_real, y_fake))

            # https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256/20
            if acc_dis.data >= (1 - self.gan_accuracy_cap):
                gen_optimizer.step()
            if acc_dis.data <= self.gan_accuracy_cap:
                dis_optimizer.step()

            if show:
                print('loss_gen:{} loss_dis:{}'.format(loss_gen, loss_dis))

            #chainer.report({'loss': loss_gen, 'z_mse': z_mse}, gen)
            #chainer.report({
            #    'loss': loss_dis, 'acc': acc_dis, 'acc/fake': acc_dis_fake,
            #    'acc/real': acc_dis_real}, dis)
