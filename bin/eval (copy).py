#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Yasunori Kudo

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from projection_gan.pose.posenet import MLP

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Generatorの重みファイルへのパス')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-b', type=int, default=200)
    parser.add_argument('--allow_inversion', action="store_true",
                         help='評価時にzの反転を許可するかどうか')
    args = parser.parse_args()

    # 学習時のオプションの読み込み
    with open(os.path.join(
            os.path.dirname(args.model_path), 'options.json')) as f:
        opts = json.load(f)

    # モデルの定義
    gen = MLP(mode='generator', use_bn=opts['use_bn'],
              activate_func=getattr(F, args.activate_func))

    gen.load_state_dict(torch.load('gen.pth'))
    gen.eval()

    # get the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gen.to(device)

    # 行動クラスの読み込み
    if opts['action'] == 'all':
        with open(os.path.join('data', 'actions.txt')) as f:
            actions = f.read().split('\n')[:-1]
    else:
        actions = [opts['action']]

    # 各行動クラスに対して平均エラー(mm)を算出
    errors = []
    for act_name in actions:
        test = projection_gan.pose.dataset.pose_dataset.H36M(
            action=act_name, length=1, train=False,
            use_sh_detection=opts['use_sh_detection'])
        test_iter = iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False)
        eds = []
        for batch in test_iter:
            xy_proj, xyz, scale = dataset.concat_examples(
                batch, device=args.gpu)
            xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                xy_real = chainer.Variable(xy_proj)
                z_pred = gen(xy_real)

            lx = gen.xp.power(xyz[:, 0::3] - xy_proj[:, 0::2], 2)
            ly = gen.xp.power(xyz[:, 1::3] - xy_proj[:, 1::2], 2)
            lz = gen.xp.power(xyz[:, 2::3] - z_pred.data, 2)

            euclidean_distance = gen.xp.sqrt(lx + ly + lz).mean(axis=1)
            euclidean_distance *= scale[:, 0]
            euclidean_distance = gen.xp.mean(euclidean_distance)

            eds.append(euclidean_distance * len(batch))
        test_iter.finalize()
        print(act_name, sum(eds) / len(test))
        errors.append(sum(eds) / len(test))
    print('-' * 20)
    print('average', sum(errors) / len(errors))

    # csvとして保存
    with open(args.model_path.replace('.npz', '.csv'), 'w') as f:
        for act_name, error in zip(actions, errors):
            f.write('{},{}\n'.format(act_name, error))
        f.write('{},{}\n'.format('average', sum(errors) / len(errors)))
