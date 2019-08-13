#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable

# define the class of CNN
# a forward CNN
class ConvNet_2(nn.Module):
    def __init__(self):
        super(ConvNet_2, self).__init__()
        # 5*100*100 - 16*96*96
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(5,16, kernel_size = 5),
            nn.BatchNorm2d(16),
            nn.PReLU(16))
        self.convlayer1 = nn.DataParallel(self.convlayer1)
        
        # NIN layer added
        self.convlayer1_1 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size = 1),
            nn.BatchNorm2d(16),
            nn.PReLU(16))
        self.convlayer1_1 = nn.DataParallel(self.convlayer1_1)
        
        # 16*96*96 - 64*48*48
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(16,64, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        self.convlayer2 = nn.DataParallel(self.convlayer2)
        # NIN layer added
        self.convlayer2_1 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size = 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        self.convlayer2_1 = nn.DataParallel(self.convlayer2_1)
        
        # 64*48*48 - 256*12*12
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size = 4, stride = 4),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        self.convlayer3 = nn.DataParallel(self.convlayer3)
        
        # NIN layer added
        self.convlayer3_1 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        self.convlayer3_1 = nn.DataParallel(self.convlayer3_1)
            
        # 256*12*12 - 256*12*12
        # res
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.Conv2d(256,256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        self.convlayer4 = nn.DataParallel(self.convlayer4)
        
        # 256*12*12 - 256*12*12
        self.convlayer5 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        self.convlayer5 = nn.DataParallel(self.convlayer5)
        
        # 256*12*12 - 256*12*12
        # res
        self.convlayer6 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.Conv2d(256,256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        self.convlayer6 = nn.DataParallel(self.convlayer6)
        
        # 256*12*12 - 256*12*12
        self.convlayer7 = nn.Sequential(
            nn.Conv2d(256,256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.PReLU(256))
        self.convlayer7 = nn.DataParallel(self.convlayer7)
        
        # 256*12*12 - 128*25*25
        self.convlayer8 = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(128),
            nn.PReLU(128))
        self.convlayer8 = nn.DataParallel(self.convlayer8)
        
        # 128*25*25 - 64*50*50
        self.convlayer9 = nn.Sequential(
            nn.ConvTranspose2d(128,64, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        self.convlayer9 = nn.DataParallel(self.convlayer9)
        
        # NIN layer
        self.convlayer9_1 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size = 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        self.convlayer9_1 = nn.DataParallel(self.convlayer9_1)
        
        # 64*50*50 - 32*100*100
        self.convlayer10 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(32),
            nn.PReLU(32))
        self.convlayer10 = nn.DataParallel(self.convlayer10)
        
        # NIN
        self.convlayer10_1 = nn.Sequential(
            nn.ConvTranspose2d(32,32, kernel_size = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32))
        self.convlayer10_1 = nn.DataParallel(self.convlayer10_1)
        
        # 32*100*100 - 12*100*100
        self.convlayer11 = nn.Sequential(
            nn.Conv2d(32,12, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(12),
            nn.PReLU(12))
        self.convlayer11 = nn.DataParallel(self.convlayer11)
        
        # 12*100*100 - 3*100*100
        self.convlayer12 = nn.Sequential(
            nn.Conv2d(12,3, kernel_size = 1),
            nn.BatchNorm2d(3),
            nn.PReLU(3))
        self.convlayer12 = nn.DataParallel(self.convlayer12)
        
    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer1_1(x)
        x = self.convlayer2(x)
        x = self.convlayer2_1(x)    
        x = self.convlayer3(x)
        x = self.convlayer3_1(x)
        x = self.convlayer4(x) + x
        x = self.convlayer5(x) + x
        x = self.convlayer6(x) + x
        x = self.convlayer7(x) + x
        x = self.convlayer8(x)
        x = self.convlayer9(x)
        x = self.convlayer9_1(x)
        x = self.convlayer10(x)
        x = self.convlayer10_1(x)
        x = self.convlayer11(x)
        x = self.convlayer12(x)
        return x

