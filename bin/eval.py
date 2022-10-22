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
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from projection_gan.pose.posenet import MLP
from projection_gan.pose.dataset.pose_dataset import H36M, MPII
from projection_gan.pose.dataset.mpii_inf_3dhp_dataset import MPII3DDataset
from projection_gan.pose.updater import H36M_Updater
from projection_gan.pose.dataset.pose_dataset_base import Normalization
#from projection_gan.pose.evaluator import Evaluator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from draw import Plotter3d, draw_poses
import cv2

import pandas as pd
import plotly.express as px

def to36M(bones, body_parts):
    H36M_JOINTS_17 = [
        'Hip',
        'RHip',
        'RKnee',
        'RFoot',
        'LHip',
        'LKnee',
        'LFoot',
        'Spine',
        'Thorax',
        'Neck/Nose',
        'Head',
        'LShoulder',
        'LElbow',
        'LWrist',
        'RShoulder',
        'RElbow',
        'RWrist',
    ]

    adjusted_bones = []
    for name in H36M_JOINTS_17:
        if not name in body_parts:
            if name == 'Hip':
                adjusted_bones.append((bones[body_parts['RHip']] + bones[body_parts['LHip']]) / 2)
            elif name == 'RFoot':
                adjusted_bones.append(bones[body_parts['RAnkle']])
            elif name == 'LFoot':
                adjusted_bones.append(bones[body_parts['LAnkle']])
            elif name == 'Spine':
                adjusted_bones.append(
                    (
                            bones[body_parts['RHip']] + bones[body_parts['LHip']]
                            + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                    ) / 4
                )
            elif name == 'Thorax':
                adjusted_bones.append(
                    (
                            + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                    ) / 2
                )
            elif name == 'Head':
                thorax = (
                                 + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                         ) / 2
                adjusted_bones.append(
                    thorax + (
                            bones[body_parts['Nose']] - thorax
                    ) * 2
                )
            elif name == 'Neck/Nose':
                adjusted_bones.append(bones[body_parts['Nose']])
            else:
                raise Exception(name)
        else:
            adjusted_bones.append(bones[body_parts[name]])

    return adjusted_bones


def parts(args):
    if args.dataset == 'COCO':
        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    else:
        assert (args.dataset == 'MPI')
        BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}

        POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
    return BODY_PARTS, POSE_PAIRS


def get_pose_trainedstyle(args, position):
    BODY_PARTS, POSE_PAIRS = parts(args)
    points = to36M(position, BODY_PARTS)
    return points    

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 3)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        print('A:{}'.format(pose_3d.shape))
        print('A:{}'.format(poses_3d[pose_id].shape))
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def centralize(pose):
    x = 0
    y = 0
    z = 0
    for i in range(0, len(pose)):
        x += pose[i][0]
        y += pose[i][1]
        z += pose[i][2]
    x /= len(pose)
    y /= len(pose)
    z /= len(pose)
    #print(x, y, z)
    r = pose[:] - [x, y, z]
    # We don't have the trajectory, but at least we can rebase the height
    r[:, 2] -= np.min(r[:, 2])    
    return r
    

def infer0_func(args, gen, device):
    #position = [948, 223, 945, 286, 860, 286, 852, 388, 843, 438, 1020, 286, 1054, 379, 1088, 421, 903, 489, 945, 603, 931, 738, 1004, 480, 1046, 596, 1054, 725, 929, 198, 961, 200, 901, 191, 987, 194 ]
    position = [(948, 223), (945, 286), (860, 286), (852, 388), (843, 438), (1020, 286), (1054, 379), (1088, 421), (903, 489), (945, 603), (931, 738), (1004, 480), (1046, 596), (1054, 725), (929, 198), (961, 200), (901, 191), (987, 194) ]
    print('#position:{}'.format(len(position)))
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for i in range(0, len(position)):
        cv2.circle(img, position[i], 2, (255, 0, 0), 2)
        img = cv2.putText(img, str(i), position[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', img)

    position = np.array(position)
    print('position:{}'.format(position))
    points = get_pose_trainedstyle(args, position)
    points = np.array(points)
    points = np.reshape(points, [1, -1]).astype('f')
    print('points:{}'.format(points))
    points = Normalization.normalize_2d(points)
    print('points:{}'.format(points))
    xy_real = torch.Tensor(points[0]).to(device)
    #z_pred = gen(xy_real) # <-- It works for the unsupervised
    z_pred = gen(xy_real.unsqueeze(0))
    x = xy_real[0::2]
    y = xy_real[1::2]
    print('#x:{}'.format(x.shape))
    print('#y:{}'.format(y.shape))
    print('#z_pred:{}'.format(z_pred.shape))
    print('#xy_real:{}'.format(xy_real.shape))
    print('xy_real:{} z_pred:{}'.format(xy_real, z_pred))
    pose = np.stack((x.cpu().detach(), y.cpu().detach(), z_pred[0].cpu().detach()), axis=-1)
    pose = np.reshape(pose, (len(x), -1))
    print('pose:{}'.format(pose))
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])    
    file_path = 'extrinsics.json'
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)

    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)
    base_height = 256
    fx = -1
    edges = []
    print('pose:{}'.format(pose.shape))
    poses_3d = pose.reshape(1, -1) * 100#np.expand_dims(pose, axis=0)
    print('poses_3d:{}'.format(poses_3d.shape))
    if len(poses_3d):
        poses_3d = rotate_poses(poses_3d, R, t)
        poses_3d_copy = poses_3d.copy()
        x = poses_3d_copy[:, 0::3]
        y = poses_3d_copy[:, 1::3]
        z = poses_3d_copy[:, 2::3]
        #poses_3d[:, 0::3], poses_3d[:, 1::3], poses_3d[:, 2::3] = -z, x, -y

        poses_3d = poses_3d.reshape(poses_3d.shape[0], 17, -1)[:, :, 0:3]
        print('#poses_3d:{}'.format(poses_3d.shape))

        edges = (Plotter3d.SKELETON_EDGES + 17 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        #print('poses_3d:{}'.format(poses_3d))
        for i in range(0, len(poses_3d)):
            poses_3d[i] = centralize(poses_3d[i])
        for i in range(0, 17):
            print('i:{} p3d:{}'.format(i, poses_3d[0][i]))
        print('poses_3d:{}'.format(poses_3d))
        #print('poses_2d:{}'.format(poses_2d))

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    # https://plotly.com/python/3d-scatter-plots/
    df = pd.DataFrame(poses_3d[0], columns=['x', 'y', 'z'])
    fig = px.scatter_3d(df, x='x', y='y', z='z')
    fig.show()    

    # Plot over the 3D image
    print('poses_3dC:{}'.format(poses_3d.shape))
    print('edgesC:{}'.format(edges.shape))
    plotter.plot(canvas_3d, poses_3d, edges)
    cv2.imshow('canvas_3d_window_name', canvas_3d)
    cv2.waitKey(0)


# PyTorch
# 948 223 945 286 860 286 852 388 843 438 1020 286 1054 379 1088 421 903 489 945 603 931 738 1004 480 1046 596 1054 725 929 198 961 200 901 191 987 194 
def infer_func(train_loader, gen, device):

    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])    
    file_path = 'extrinsics.json'
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
        
    iter = 0
    for iterations in range(0, 1):
        print('iterations:{}'.format(iterations))
        for res in train_loader:
            xy_proj, xyz, scale = torch.squeeze(res[0]), torch.squeeze(res[1]), torch.squeeze(res[2])
            batchsize = xy_proj.shape[0]
            show = False
            if iter % 100 == 0: show = True
            iter += 1

            if show:
                print('iter:{}'.format(iter))
            print('xy_proj:{}'.format(xy_proj))
            xy_real = torch.Tensor(xy_proj).to(device)
            #z_pred = gen(xy_real)  # <-- It works with the unsupervised
            z_pred = gen(xy_real.unsqueeze(0))
            x = xy_real[0::2]
            y = xy_real[1::2]
            print('#x:{}'.format(x.shape))
            print('#y:{}'.format(y.shape))
            print('#z_pred:{}'.format(z_pred.shape))
            print('#xy_real:{}'.format(xy_real.shape))
            print('xy_real:{} z_pred:{}'.format(xy_real, z_pred))
            pose = np.stack((x.cpu().detach(), y.cpu().detach(), z_pred[0].cpu().detach()), axis=-1)
            pose = np.reshape(pose, (len(x), -1))
            print('pose:{}'.format(pose))
            canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
            plotter = Plotter3d(canvas_3d.shape[:2])    
            file_path = 'extrinsics.json'
            with open(file_path, 'r') as f:
                extrinsics = json.load(f)

            R = np.array(extrinsics['R'], dtype=np.float32)
            t = np.array(extrinsics['t'], dtype=np.float32)
            base_height = 256
            fx = -1
            edges = []
            print('pose:{}'.format(pose.shape))
            poses_3d = pose.reshape(1, -1) * 100#np.expand_dims(pose, axis=0)
            print('poses_3d:{}'.format(poses_3d.shape))
            if len(poses_3d):
                poses_3d = rotate_poses(poses_3d, R, t)
                poses_3d_copy = poses_3d.copy()
                x = poses_3d_copy[:, 0::3]
                y = poses_3d_copy[:, 1::3]
                z = poses_3d_copy[:, 2::3]
                #poses_3d[:, 0::3], poses_3d[:, 1::3], poses_3d[:, 2::3] = -z, x, -y

                poses_3d = poses_3d.reshape(poses_3d.shape[0], 17, -1)[:, :, 0:3]
                print('#poses_3d:{}'.format(poses_3d.shape))

                edges = (Plotter3d.SKELETON_EDGES + 17 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
                #print('poses_3d:{}'.format(poses_3d))
                for i in range(0, len(poses_3d)):
                    poses_3d[i] = centralize(poses_3d[i])
                for i in range(0, 17):
                    print('i:{} p3d:{}'.format(i, poses_3d[0][i]))
                print('poses_3d:{}'.format(poses_3d))
                #print('poses_2d:{}'.format(poses_2d))

            # Plot over the 3D image
            print('poses_3dC:{}'.format(poses_3d.shape))
            print('edgesC:{}'.format(edges.shape))
            plotter.plot(canvas_3d, poses_3d, edges)
            cv2.imshow('canvas_3d_window_name', canvas_3d)
            cv2.waitKey(0)

def test_plotly():
    df = px.data.iris()
    print('df:{}'.format(type(df)))
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                color='species')
    fig.show()    

def main():
    #test_plotly()
    #exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-b', '--batchsize', type=int, default=1)
    parser.add_argument('-B', '--test_batchsize', type=int, default=1)
    parser.add_argument('-r', '--resume', default='')
    parser.add_argument('-o', '--out', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-m', '--mode', type=str, default='unsupervised',
                        choices=['supervised', 'unsupervised'])
    parser.add_argument('-d', '--dataset', type=str, default='h36m',
                        choices=['COCO', 'MPI'])
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

    # get the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Models
    gen = MLP(mode='generator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))
    gen.to(device)
    gen.load_state_dict(torch.load('gen.pth'))
    gen.eval()

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

    # train the model
    #infer0_func(args, gen, device)
    infer_func(test_loader, gen, device)

if __name__ == '__main__':
    main()
