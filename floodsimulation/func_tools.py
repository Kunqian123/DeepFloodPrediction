
# coding: utf-8

# In[ ]:


import os
import time
import sys

# Related major packages
import anuga

# Application specific imports              # Definition of file names and polygons
import aw_small_project

from numpy import allclose
import numpy as np
from math import sin
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
from anuga.utilities.numerical_tools import ensure_numeric
from anuga.geospatial_data.geospatial_data import ensure_absolute

from csv import reader,writer
from anuga.utilities.numerical_tools import ensure_numeric, mean, NAN
import string
from anuga.utilities.file_utils import get_all_swwfiles
from anuga.abstract_2d_finite_volumes.util import file_function  
from anuga.geospatial_data.geospatial_data import ensure_absolute
import gc
from csv import reader,writer
from anuga.utilities.numerical_tools import ensure_numeric, mean, NAN
import string
from anuga.utilities.file_utils import get_all_swwfiles
from anuga.abstract_2d_finite_volumes.util import file_function  
from anuga.geospatial_data.geospatial_data import ensure_absolute
from anuga.geospatial_data.geospatial_data import ensure_absolute
from anuga.file.netcdf import NetCDFFile
from anuga.config import netcdf_mode_r, netcdf_mode_w, netcdf_mode_a
from anuga.utilities.numerical_tools import ensure_numeric
from anuga.fit_interpolate.interpolate import Interpolate
from anuga.abstract_2d_finite_volumes.neighbour_mesh import Mesh
from anuga.geometry.polygon import outside_polygon

def get_matrix_A(fn, gauge_name):
    # points to read information from
    point_reader = reader(file(gauge_name))
    points = []
    point_name = []
    for i,row in enumerate(point_reader):
        if i==0:
            for j,value in enumerate(row):
                if value.strip()=='easting':easting=j
                if value.strip()=='northing':northing=j
                if value.strip()=='name':name=j
                if value.strip()=='elevation':elevation=j
        else:
            #points.append([float(row[easting]),float(row[northing])])
            points.append([float(row[easting]),float(row[northing])]) 
            point_name.append(row[name])
    points_array = np.array(points,np.float)
    dim_gauge = int(np.sqrt(points_array.shape[0]))
    interpolation_points = ensure_absolute(points_array)
    
    # read the sww file to extract something
    fid = NetCDFFile(fn, netcdf_mode_r)
    xllcorner = fid.xllcorner
    yllcorner = fid.yllcorner
    zone = fid.zone
    x = fid.variables['x'][:]
    y = fid.variables['y'][:]

    triangles = fid.variables['volumes'][:]

    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    vertex_coordinates = np.concatenate((x, y), axis=1)
    ele = fid.variables['elevation'][:]
    fid.close()
    

    vertex_coordinates = ensure_absolute(vertex_coordinates)
    triangles = ensure_numeric(triangles)

    interpolation_points = ensure_numeric(interpolation_points)
    interpolation_points[:, 0] -= xllcorner
    interpolation_points[:, 1] -= yllcorner  
    
    mesh = Mesh(vertex_coordinates, triangles)
    mesh_boundary_polygon = mesh.get_boundary_polygon()
    indices = outside_polygon(interpolation_points,
                           mesh_boundary_polygon)
    interp = Interpolate(vertex_coordinates, triangles)
    matrix_A, inside_poly_indices, outside_poly_indices, centroids = interp._build_interpolation_matrix_A(interpolation_points,output_centroids=False,verbose=False)
    
    ele = matrix_A*ele
    ele = np.asarray(ele)
    ele = ele.reshape((100,100))
    
    return ele, matrix_A

def read_result(fn, save_pl, matrix_A, ele, quantity_names, case_num ,time_thinning = 1):
    fid = NetCDFFile(fn, netcdf_mode_r)
    time = fid.variables['time'][:]
    upper_time_index = len(time)  
    quantities = {}
    for i, name in enumerate(quantity_names):
        quantities[name] = fid.variables[name][:]
        quantities[name] = np.array(quantities[name][::time_thinning,:])
    fid.close()
    for i, t in enumerate(time):
        for name in quantity_names:
            Q = quantities[name][i,:] # Quantities at timestep i
            result = matrix_A*Q
            result = np.asarray(result)
            result = result.reshape((100,100))
            if name == 'stage':
                result = result - ele
                save_name = save_pl + case_num + '_' + 'depth' + '_{0}.csv'.format(i)
            else:
                save_name = save_pl + case_num + '_' + name + '_{0}.csv'.format(i)
            np.savetxt(save_name, result, delimiter = ",")
            
def input_matrix(case, time):
    # inflow10: the flow out of dam
    # outflow1: the flow 
    input_name = "E:/austinflood_sww/flood_result_boundaryinflow/input" + case + ".csv"
    input_ = pd.read_csv(input_name)
    input_ = np.asarray(input_)
    
    rain1 = input_[time,0]/(455000000.0/9)
    rain2 = input_[time,1]/(455000000.0/9)
    rain3 = input_[time,2]/(455000000.0/9)
    rain4 = input_[time,3]/(455000000.0/9)
    rain5 = input_[time,4]/(455000000.0/9)
    rain6 = input_[time,5]/(455000000.0/9)
    rain7 = input_[time,6]/(455000000.0/9)
    rain8 = input_[time,7]/(455000000.0/9)
    rain9 = input_[time,8]/(455000000.0/9)
    
    inflow1 = input_[time,9]/(3.14*80*80)
    inflow2 = input_[time,10]/(3.14*120*120)
    inflow3 = input_[time,11]/(3.14*120*120)
    inflow4 = input_[time,12]/(3.14*120*120)
    inflow5 = input_[time,13]/(3.14*120*120)
    inflow10 = input_[time,14]/(3.14*120*120)
    outflow1 = input_[time,15]/(3.14*120*120)
    
    matrix2 = np.zeros((100,100))
    matrix = np.zeros((100,100))
    matrix[0:33,0:33] = matrix[0:33,0:33] + rain1
    matrix[33:67,0:33] = matrix[33:67,0:33] + rain2
    matrix[67:100,0:33] = matrix[67:100,0:33] + rain3
    matrix[0:33,33:67] = matrix[0:33,33:67] + rain4
    matrix[33:67,33:67] = matrix[33:67,33:67] + rain5
    matrix[67:100,33:67] = matrix[67:100,33:67] + rain6
    matrix[0:33,67:100] = matrix[0:33,67:100] + rain7
    matrix[33:67,67:100] = matrix[33:67,67:100] + rain8
    matrix[67:100,67:100] = matrix[67:100,67:100] + rain9
    
    matrix2[3:5, 90:92] = matrix2[3:5, 90:92] + inflow1
    matrix2[12:14, 96:98] = matrix2[12:14, 96:98] + inflow2
    matrix2[8:10, 68:70] = matrix2[8:10, 68:70] + inflow3
    matrix2[13:15, 41:43] = matrix2[13:15, 41:43] + inflow4
    matrix2[83:86, 6:8] = matrix2[83:86, 6:8] + inflow5
    matrix2[13:15, 61:63] = matrix2[13:15, 61:63] + inflow10
    matrix2[14:16, 62:64] = matrix2[14:16, 62:64] + outflow1
    
    rain_fn = saving_file + 'rain' + case + '.csv'
    flow_fn = saving_file + 'inflow' + case + '.csv'
    #rain_fn = 'E:/austinflood_sww/rain_{0}_{1}.csv'.format(case, time)
    #flow_fn = 'E:/austinflood_sww/flow_{0}_{1}.csv'.format(case, time)
    np.savetxt(rain_fn, matrix, delimiter = ",")
    np.savetxt(flow_fn, matrix2, delimiter = ",")

