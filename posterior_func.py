#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import torchvision.utils as vutils
from torch.autograd import Variable

import models.UNet as net

import models.UNet as net_test
from misc import *
from torchsummary import summary
import datetime
import torch.nn.functional as F
import torch.optim as optim
import math
import pandas as pd
import numpy as np
import random

from basic_toolfunc import normalization_test, local_area, coord_trans, number_to_coord
from model_test_func import prior_update_gan_noise
from models.gan_models import G, D

class ConvNet_nin(nn.Module):
    def __init__(self):
        super(ConvNet_nin, self).__init__()
        # 5*14*14 - 16*12*12
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(5,196, kernel_size = 16),
            nn.BatchNorm2d(196),
            nn.PReLU(196))
        #self.convlayer1 = nn.DataParallel(self.convlayer1)
        
#         # 32*12*12 - 64 * 8 * 8
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(196,24, kernel_size = 12),
            nn.BatchNorm2d(24),
            nn.PReLU(24))
#         self.convlayer3 = nn.DataParallel(self.convlayer3)
        
        # 64*8*8 - 32 * 6 * 6
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(24,3, kernel_size = 4))
    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        return x
    
def prior_update_gan_noise(particles, inputs, model_no):
    netG = net.G(5, 3, 64)
    netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % ('/home/gtx1080/Abduallah/pix2pix.pytorch/imglog/fluid_noise', model_no)))
    netG.eval()
    netG.cuda()
    mu = np.load('/home/gtx1080/Sync/Kun/30_min_ele/non_ele_whole/train/mu.npy')
    var = np.load('/home/gtx1080/Sync/Kun/30_min_ele/non_ele_whole/train/var.npy')
    particles = np.concatenate((particles, inputs), axis = 1)
    particles, label = normalization_test(particles, particles[:,0:3,:,:], mu, var)
    particles = torch.tensor(particles, dtype = torch.float).cuda()
    with torch.no_grad():
        particles = netG(particles)
    particles = np.array(particles.cpu())
    particles[:,0,:,:] = particles[:,0,:,:]*np.sqrt(var[5,0]) + mu[5,0]
    particles[:,1,:,:] = particles[:,1,:,:]*np.sqrt(var[6,0]) + mu[6,0]
    particles[:,2,:,:] = particles[:,2,:,:]*np.sqrt(var[7,0]) + mu[7,0]
    particles[:,0,:,:] = np.maximum(particles[:,0,:,:], 0)
    return particles

def ensemble_learning(states, inputs, gan_no, posgan_no, index_list, point_list_2, cov, obs_var):
    # states: present states shape N*3*100*100
    # inputs: inputs in the future 30 minutes shape N*2*100*100
    # gan_no: the number of trained gan models
    # index_list: 
    # point_list_2: point_list_2
    # cov: covariance matrix used: test_cov
    # obs_var: the variance of error: [0.000001, 0.0001, 0.0001]
    cnnpredictions = meas_data_read(states, inputs, index_list)
    ganpredictions = prior_update_gan_noise(states, inputs, gan_no)
    cnnpredictions = np.array(cnnpredictions)
    
    obs_value_list = obs_list_form_hxy(point_list_2, cnnpredictions)
    posgan_prediction = two_model_merge(ganpredictions, obs_var, point_list_2, cov, obs_value_list)
    posgan_prediction = posgan_prediction.reshape((-1, 3,100,100))
    
    print(posgan_prediction.shape)
    print(inputs.shape)
    en_pred_input = np.concatenate((posgan_prediction, inputs), axis = 1)
    posposgan_prediction = pos_ensemble_gan_noise(en_pred_input, posgan_no)
    
    return ganpredictions, posgan_prediction, posposgan_prediction

# data assimilation functions
def meas_data_read(state, inputs, point_list):
    # first do an update with states and input of specific points
    # inputs: state, input, the list of points
    # return the list of values of specific points
    particles = np.concatenate((state, inputs), axis = 1)
    
    time_series_padded = np.zeros(shape = (particles.shape[0],particles.shape[1],130,130))
    time_series_padded[:,:,15:115,15:115] = particles

    prediction = []
    label = np.zeros((particles.shape[0],3,100,100))

    for ele in point_list:
#         print(ele)
        [index_x, index_y] = ele
#         print(index_x)
#         print(index_y)
#             print((index_x, index_y))

        # temp list to save errors
        mu_fn = '/home/gtx1080/Sync/Kun/30_min_ele/boundary_para/mu_{0}_{1}.npy'.format(index_x, index_y)
        var_fn = '/home/gtx1080/Sync/Kun/30_min_ele/boundary_para/var_{0}_{1}.npy'.format(index_x, index_y)
        mu = np.load(mu_fn)
        var = np.load(var_fn)

        # load the first model
        #del model_1
        model = torch.load('/home/gtx1080/Sync/Kun/30_min_ele/trained_boundary/trainedmodel_noised_{0}_{1}'.format(index_x, index_y))
        model.eval()

        # take out the local data
        time_series_input_ = time_series_padded[ :, :, index_x+15-15:index_x+15+15, index_y+15-15:index_y+15+15]
#             time_series_label_ = time_series[ :, :, index_x, index_y]
#             time_series_label_ = time_series_label_.reshape((time_series_label_.shape[0], 3, 1, 1))

        # time_series_input_, time_series_label_  = normalization_test(time_series_input_, time_series_label_, test_mu, test_var)
        time_series_input_, time_series_label_  = normalization_test(time_series_input_, time_series_input_[:,0:3,:,:], mu, var)
        time_series_input_ = torch.tensor(time_series_input_, dtype = torch.float)
        time_series_input_ = time_series_input_.cuda()

        # del model
        model.eval()
        with torch.no_grad():
            test_prediction = model.forward(time_series_input_)
        test_prediction = test_prediction.cpu()
        test_prediction = np.array(test_prediction)

        test_prediction[:,0,:,:] = test_prediction[:,0,:,:]*np.sqrt(var[5,0]) + mu[5,0]
        test_prediction[:,1,:,:] = test_prediction[:,1,:,:]*np.sqrt(var[6,0]) + mu[6,0]
        test_prediction[:,2,:,:] = test_prediction[:,2,:,:]*np.sqrt(var[7,0]) + mu[7,0]

#         prediction[:,:,index_x, index_y] = test_prediction[:,:,0,0]
        prediction.append(test_prediction[:,:,0,0])
    
    return prediction

def obs_list_form(point_list, prediction_array):
    # reform the result from function:meas_data_read() to the input to ensemble function
    # prediction_array the result form last function (predictions)
    # point_list: the list of points(point_list_2)
    obs_value_list = []
    count = 0
    for ele in point_list:
        test_x, test_y, label = ele
        temp_i = int(count)
#         print(temp_i)
        if label=='h':
            measurement_value = prediction_array[temp_i, :, 0]
            obs_value_list.append(measurement_value.copy())
        if label=='x':
            measurement_value = prediction_array[temp_i, :, 1]
            obs_value_list.append(measurement_value.copy())
        if label=='y':
            measurement_value = prediction_array[temp_i, :, 2]
            obs_value_list.append(measurement_value.copy())
        count = count + 1
    obs_value_list = np.array(obs_value_list)
    print(obs_value_list.shape)
    obs_value_list = obs_value_list.T.reshape((prediction_array.shape[1],260,-1))
    print(prediction_array.shape)
    print(obs_value_list.shape)
    return obs_value_list

def obs_list_form_hxy(point_list, prediction_array):
    # prediction_array the result form last function (predictions)
    # point_list: the list of points(point_list_2)
#     dim = len(point_list)
    print(len(point_list))
    print(prediction_array.shape)
    obs_value_list = []
    count = 0
    for ele in point_list:
        test_x, test_y, label = ele
        temp_i = int(count/3)
#         print(temp_i)
        if label=='h':
            measurement_value = prediction_array[temp_i, :, 0]
            obs_value_list.append(measurement_value.copy())
        if label=='x':
            measurement_value = prediction_array[temp_i, :, 1]
            obs_value_list.append(measurement_value.copy())
        if label=='y':
            measurement_value = prediction_array[temp_i, :, 2]
            obs_value_list.append(measurement_value.copy())
        count = count + 1
    obs_value_list = np.array(obs_value_list)
#     print(obs_value_list.shape)
    obs_value_list = obs_value_list.T.reshape((prediction_array.shape[1],3,-1))
#     print(prediction_array.shape)
#     print(obs_value_list.shape)
    return obs_value_list

# the function conduct the posterior update
def two_model_merge(temp_gan_prediction, obs_var_list, point_list, test_cov1, obs_value_list):
    # temp_gan_prediction: gan_prediction
    # obs_var_list = [0.00001, 0.001, 0.001]
    # point_list: point_list_2
    # test_cov1: test_cov1
    # obs_value_list: obs_value_list
    gan_shape = temp_gan_prediction.shape
    n_sample = gan_shape[0]
    obs_num = len(point_list)
    obs_ = np.zeros((obs_num,30000))
    for i in range(obs_num):
        obs_x, obs_y, label = point_list[i]
        if label=='h':
            index_ = coord_trans(obs_x, obs_y)
            obs_[i,index_] = 1.0
        if label=='x':
            index_ = coord_trans(obs_x, obs_y) + 10000
            obs_[i,index_] = 1.0
        if label=='y':
            index_ = coord_trans(obs_x, obs_y) + 20000
            obs_[i,index_] = 1.0

    obs_var = np.zeros((obs_num, obs_num))

    for i in range(obs_num):
        if point_list[i][2] == 'h':
            obs_var[i,i] = obs_var_list[0]
        if point_list[i][2] == 'x':
            obs_var[i,i] = obs_var_list[1]
        if point_list[i][2] == 'y':
            obs_var[i,i] = obs_var_list[2]
    temp_gan_prediction = temp_gan_prediction.reshape((n_sample, 30000)).T
    obs_value_list = obs_value_list.reshape((n_sample,obs_num)).T
    posgan_prediction = temp_gan_prediction + test_cov1.dot(obs_.T).dot(np.linalg.inv(obs_.dot(test_cov1).dot(obs_.T) + obs_var)).dot(obs_value_list - np.dot(obs_, temp_gan_prediction))
    return posgan_prediction

def pos_ensemble_gan_noise(inputs, model_no):
    netG = net.G(5, 3, 64)
    netG.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % ('/home/gtx1080/Abduallah/pix2pix.pytorch/sample', model_no)))
    netG.eval()
    netG.cuda()
    mu = np.load('/home/gtx1080/Sync/Kun/30_min_ele/non_ele_whole/train/mu.npy')
    var = np.load('/home/gtx1080/Sync/Kun/30_min_ele/non_ele_whole/train/var.npy')
#     particles = np.concatenate((particles, inputs), axis = 1)
    inputs, label = normalization_test(inputs, inputs[:,0:3,:,:], mu, var)
    inputs = torch.tensor(inputs, dtype = torch.float).cuda()
    with torch.no_grad():
        inputs = netG(inputs)
    inputs = np.array(inputs.cpu())
    inputs[:,0,:,:] = inputs[:,0,:,:]*np.sqrt(var[5,0]) + mu[5,0]
    inputs[:,1,:,:] = inputs[:,1,:,:]*np.sqrt(var[6,0]) + mu[6,0]
    inputs[:,2,:,:] = inputs[:,2,:,:]*np.sqrt(var[7,0]) + mu[7,0]
    inputs[:,0,:,:] = np.maximum(inputs[:,0,:,:], 0)
    return inputs

