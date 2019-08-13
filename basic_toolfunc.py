#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def normalization_test(test_input_, test_label_, mu, var):
    test_input_normalized = np.zeros(shape = test_input_.shape)
    test_label_normalized = np.zeros(shape = test_label_.shape)
    for i in range(test_input_.shape[1]):
        if var[i] != 0:
            test_input_normalized[:,i,:,:] = (test_input_[:,i,:,:] - mu[i]) / np.sqrt(var[i])
        else:
            test_input_normalized[:,i,:,:] = (test_input_[:,i,:,:] - mu[i])
    for i in range(test_input_.shape[1], test_input_.shape[1] + test_label_.shape[1]):
        if var[i] != 0:
            test_label_normalized[:,i - test_input_.shape[1],:,:] = (test_label_[:,i- test_input_.shape[1],:,:] - mu[i]) / np.sqrt(var[i])
        else:
            test_label_normalized[:,i - test_input_.shape[1],:,:] = (test_label_[:,i- test_input_.shape[1],:,:] - mu[i])
    return test_input_normalized, test_label_normalized

def local_area(data_input, data_label, index_x, index_y):
    # data_input: the whole matrix of size n*5*114*114
    # data_label: the whole matrix of size n*3*114*114
    # index_x, index_y: the center of extracted local area
    data_input_ = data_input[:,:,(index_x - 15):(index_x + 15), (index_y - 15):(index_y + 15)].copy()
    data_label_ = data_label[:,:,index_x ,index_y ].copy()
    data_label_ = data_label_.reshape((-1,3,1,1))
    return data_input_, data_label_

def coord_trans(x,y,channel = 'h'):
    # transform from 2D location to the 1D state space loccation
    if channel == 'h':
        index_ = x * 100 + y
    if channel == 'x':
        index_ = 100 * 100 + x * 100 + y
    if channel == 'y':
        index_ = 2 * 100 * 100 + x * 100 + y
    return index_

def number_to_coord(i):
    y = int(np.mod(i,100))
    x = int(i/100)
    return [x, y]

