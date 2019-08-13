#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from torchsummary import summary
import datetime
from random import randrange, uniform

class ConcatAndPad2d(nn.Module):
    def __init__(self):
        super(ConcatAndPad2d,self).__init__()
    def forward(self,x1,x2,dim=1): #dim =1 is the channel location
        diff_w = abs(x1.size()[2] - x2.size()[2])
        diff_h = abs(x1.size()[3] - x2.size()[3])
        
        if diff_w == diff_h and diff_w == 0: 
            return torch.cat([x1, x2], dim=dim) 
        if diff_h !=0: 
            if x1.size()[3] < x2.size()[3]: 
                x1 = F.pad(x1, (diff_h,0,0,0), "constant", 0)
            else:
                x2 = F.pad(x2, (diff_h,0,0,0), "constant", 0)
        if diff_w !=0: 
            if x1.size()[2] < x2.size()[2]: 
                x1 = F.pad(x1, (0,0,diff_w,0), "constant", 0)
            else:
                x2 = F.pad(x2, (0,0,diff_w,0), "constant", 0)

        return torch.cat([x1, x2], dim=dim)  

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False,spe=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % (name), nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % (name), nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        if not spe:
            block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
        else:
            block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 4, 1, bias=False,output_padding =2))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class D(nn.Module):
    def __init__(self, nc, nf):
        super(D, self).__init__()

        main = nn.Sequential()
        # 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

        # 128
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 64
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 32    
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
        main.add_module('%s_bn' % name, nn.BatchNorm2d(nf*2))

        # 31
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
        main.add_module('%s_sigmoid' % name , nn.Sigmoid())
        # 30 (sizePatchGAN=30)

        self.main = main

    def forward(self, x):
        output = self.main(x)
        return output


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
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

