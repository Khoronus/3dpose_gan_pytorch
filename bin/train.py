#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

from __future__ import print_function
import argparse
from asyncio import DatagramProtocol
import json
import multiprocessing
import time

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from projection_gan.pose.posenet import MLP
from projection_gan.pose.dataset.pose_dataset import H36M, MPII
from projection_gan.pose.dataset.mpii_inf_3dhp_dataset import MPII3DDataset
from projection_gan.pose.updater import H36M_Updater
#from projection_gan.pose.evaluator import Evaluator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions


def create_result_dir(dirname):
    if not os.path.exists('results'):
        os.mkdir('results')
    if dirname:
        result_dir = os.path.join('results', dirname)
    else:
        result_dir = os.path.join(
            'results', time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return result_dir


# PyTorch
def train_func(manager, model, optimizer, device, train_loader):
    while not manager.stop_trigger:
        model.train()
        for x, t in train_loader:
            with manager.run_iteration():
                x, t = x.to(device), t.to(device)
                optimizer.zero_grad()
                loss = model(x, t)
                ppe.reporting.report({'main/loss': loss.item()})
                ppe.reporting.report({'main/accuracy': model.accuracy.item()})
                loss.backward()
                optimizer.step()



# PyTorch
def train_func(updater, train_loader, gen, dis, gen_optimizer, dis_optimizer, device):
    iter = 0
    for iterations in range(0, 100):
        print('iterations:{}'.format(iterations))
        for res in train_loader:
            #print('res:{}'.format(res))
            #print('res:{}'.format(len(res)))
            #for i in range(0, len(res)):
            #    print('i:{} l:{}'.format(i, res[i].shape))
            xy_proj, xyz, scale = torch.squeeze(res[0]), torch.squeeze(res[1]), torch.squeeze(res[2])
            #print('xzy:{} {} {}'.format(xy_proj.shape, xyz.shape, scale.shape))

            batchsize = xy_proj.shape[0]
            #print('batchsize:{}'.format(batchsize))

            show = False
            if iter % 100 == 0: show = True
            iter += 1

            if show:
                print('iter:{}'.format(iter))

            updater.update_core(batchsize, xy_proj, xyz, scale, gen, dis, gen_optimizer, dis_optimizer, device, show)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-b', '--batchsize', type=int, default=16)
    parser.add_argument('-B', '--test_batchsize', type=int, default=32)
    parser.add_argument('-r', '--resume', default='')
    parser.add_argument('-o', '--out', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-m', '--mode', type=str, default='unsupervised',
                        choices=['supervised', 'unsupervised'])
    parser.add_argument('-d', '--dataset', type=str, default='h36m',
                        choices=['h36m', 'mpii', 'mpi_inf'])
    parser.add_argument('-a', '--activate_func',
                        type=str, default='leaky_relu')
    parser.add_argument('-c', '--gan_accuracy_cap', type=float, default=0.9,
                        help="Disのaccuracyがこれを超えると更新しない手加減")
    parser.add_argument('-A', '--action', type=str, default='all')
    parser.add_argument('-s', '--snapshot_interval', type=int, default=1)
    parser.add_argument('-l', '--log_interval', type=int, default=1)
    parser.add_argument('--heuristic_loss_weight', type=float, default=1.0)
    parser.add_argument('--use_heuristic_loss', action="store_true")
    parser.add_argument('--use_sh_detection', action="store_true")
    parser.add_argument('--use_bn', action="store_true")
    args = parser.parse_args()
    args.out = create_result_dir(args.out)

    # Save options.
    with open(os.path.join(args.out, 'options.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps(vars(args), indent=2))

    # get the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Models
    gen = MLP(mode='generator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))
    dis = MLP(mode='discriminator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))
    gen.to(device)
    dis.to(device)

    # Setup an optimizer
    def make_optimizer(model):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load dataset.
    if args.dataset == 'h36m':
        train = H36M(action=args.action, length=1, train=True,
                     use_sh_detection=args.use_sh_detection)
        test = H36M(action=args.action, length=1, train=False,
                    use_sh_detection=args.use_sh_detection)
    elif args.dataset == 'mpii':
        train = MPII(train=True, use_sh_detection=args.use_sh_detection)
        test = MPII(train=False, use_sh_detection=args.use_sh_detection)
    elif args.dataset == 'mpi_inf':
        train = MPII3DDataset(train=True)
        test = MPII3DDataset(train=False)
    print('TRAIN: {}, TEST: {}'.format(len(train), len(test)))

    multiprocessing.set_start_method('spawn')
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.test_batchsize, shuffle=True)

    # Set up a trainer
    updater = H36M_Updater(
        gan_accuracy_cap=args.gan_accuracy_cap,
        use_heuristic_loss=args.use_heuristic_loss,
        heuristic_loss_weight=args.heuristic_loss_weight,
        mode=args.mode)

    # train the model
    train_func(updater, train_loader, gen, dis, opt_gen, opt_dis, device)

    # save the models
    torch.save(gen.state_dict(), 'gen.pth')
    torch.save(dis.state_dict(), 'dis.pth')

if __name__ == '__main__':
    main()
