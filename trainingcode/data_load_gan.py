#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from torchsummary import summary
import datetime
from random import randrange, uniform
import matplotlib.pyplot as plt
from matplotlib import cm

def index_group(batch_num, train_num, test_num):
    # function to group the data into groups according to the No. of pieces
    # input
    # batch_num: the number of data in a batch
    # train_num: total amount of training data
    # test_num: total amount of test data
    # output
    # two lists of groups of batch number
    p = list(np.random.permutation(train_num))
    q = [ele + train_num for ele in list(np.random.permutation(test_num))]
    test_group = q[0:batch_num]
    train_group = []
    group_num = int(train_num/batch_num)
    for i in range(group_num):
        train_group.append(p[batch_num*i: batch_num*i+batch_num])
    return train_group, test_group

def data_loading(index_group, path):
    # this is the function that loads data according to the batch groups
    # input: batch group which is the output from index_group
    # output: the batch of input and batch of output ready for training
    data_ip = []
    data_op = []
    batch_size = len(index_group)
    for i in index_group:
        path_ip = path + str(i) + '_inp.npy'
        path_op = path + str(i) + '_op.npy'
        temp_ip = np.load(path_ip)
        temp_op = np.load(path_op)
        temp_ip = temp_ip.reshape((10000, -1)).T
        temp_op = temp_op.reshape((10000, -1)).T
        data_ip.append(temp_ip.copy())
        data_op.append(temp_op.copy())
    data_ip = np.array(data_ip)
    data_op = np.array(data_op)
    data_ip = data_ip.reshape((batch_size, -1, 100, 100))
    data_op = data_op.reshape((batch_size, -1, 100, 100))
    return data_ip, data_op

