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
import torch.optim as optim
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
import numpy as np

from basic_toolfunc import normalization_test, local_area, coord_trans, number_to_coord
from models.CNN_models import ConvNet_2
from models.gan_models import G, D

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

def CNN_update(particles, inputs, model_no):
    model = G()
    model = torch.load('/home/gtx1080/Sync/Kun/0614CNN_Res/{0}_model'.format(model_no))
    model.eval()
    model.cuda()
    mu = np.load('/home/gtx1080/Sync/Kun/30_min_ele/non_ele_whole/train/mu.npy')
    var = np.load('/home/gtx1080/Sync/Kun/30_min_ele/non_ele_whole/train/var.npy')
    particles = np.concatenate((particles, inputs), axis = 1)
    particles, label = normalization_test(particles, particles[:,0:3,:,:], mu, var)
    particles = torch.tensor(particles, dtype = torch.float).cuda()
    with torch.no_grad():
        particles = model(particles)
    particles = np.array(particles.cpu())
    particles[:,0,:,:] = particles[:,0,:,:]*np.sqrt(var[5,0]) + mu[5,0]
    particles[:,1,:,:] = particles[:,1,:,:]*np.sqrt(var[6,0]) + mu[6,0]
    particles[:,2,:,:] = particles[:,2,:,:]*np.sqrt(var[7,0]) + mu[7,0]
    particles[:,0,:,:] = np.maximum(particles[:,0,:,:], 0)
    return particles

# the function to test the GAN models without poeterior update step
def CNNGAN_update_path(particles, inputs, model_no):
    # employed the model 1
    model = G()
    model.load_state_dict(torch.load('trainedmodels/0718_newGAN/{0}_modelG.pth'.format(model_no)))
    model.eval()
    model.cuda()
    mu = np.load('test_code/mu.npy')
    var = np.load('test_code/var.npy')
    particles = np.concatenate((particles, inputs), axis = 1)
    particles, label = normalization_test(particles, particles[:,0:3,:,:], mu, var)
    particles = torch.tensor(particles, dtype = torch.float).cuda()
    with torch.no_grad():
        particles = model(particles)
    particles = np.array(particles.cpu())
    particles[:,0,:,:] = particles[:,0,:,:]*np.sqrt(var[5,0]) + mu[5,0]
    particles[:,1,:,:] = particles[:,1,:,:]*np.sqrt(var[6,0]) + mu[6,0]
    particles[:,2,:,:] = particles[:,2,:,:]*np.sqrt(var[7,0]) + mu[7,0]
    particles[:,0,:,:] = np.maximum(particles[:,0,:,:], 0)
    return particles

