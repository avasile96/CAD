# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:34:02 2021

@author: Manuel Ojeda & Alexandru Vasile
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import gc
import cv2
from libb import preprocessing
from skimage.morphology import area_opening

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = "D:\\Uni\\Spain\\CADx\\test"

# Generator Parameters
img_size = (450, 600,3) # RGB imges!
batch_size = 32
kernel = np.ones((5,5),np.uint8)


# Getting test images
y_train = []
x_train = []
for case in os.listdir(os.path.join(dataset_dir)):
        if case.endswith(".jpg") and not case.startswith("."):
            pseudo_x = cv2.imread(os.path.join(dataset_dir, case))
            x_train.append(pseudo_x)
            y_train.append(case)

del pseudo_x              
gc.collect()
    
#%% Preprocessing & Feature Extraction
# List to array
x_train_arr = np.array(x_train)


### Color Space Transformation: RGB --> HSV ###
Y_val = []
Y_train = []
mean_of_train_hue = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_train_sat = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_train_val = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T

mean_of_val_hue = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_val_sat = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T
mean_of_val_val = np.zeros(x_train_arr.shape[0], dtype = np.float32)[np.newaxis].T


for i in range(0,x_train_arr.shape[0]):
    x_train_arr[i] = cv2.cvtColor(x_train_arr[i],cv2.COLOR_RGB2HSV)
    x_train_arr[i] = preprocessing(x_train_arr[i])
    print(i)
    # more segemntation
    x_l_mask = np.array(x_train_arr[i,:,:,2] > 0, dtype = np.uint8)
    x_up_mask = np.array(x_train_arr[i,:,:,2] < 200 , dtype = np.uint8)
    x_mask = np.multiply(x_l_mask, x_up_mask)
    x_mask_open = area_opening(np.uint8(x_mask))
    dil_img = cv2.dilate(x_mask_open,kernel,iterations = 3)
    for j in range(0,x_train_arr[i].shape[2]):
        x_train_arr[i,:,:,j] = np.multiply(dil_img, x_train_arr[i,:,:,j])
    
    Y_train.append(y_train[i])
    
    filename_train = 'D:\\Uni\\Spain\\CADx\\preprocessing_test\\{}_{}.jpg'.format(y_train[i],i)
    cv2.imwrite(filename_train,x_train_arr[i,:,:,2])
    
    mean_of_train_hue[i] = np.mean(x_train_arr[i,:,:,0]) # getting the mean of the hue channel
    mean_of_train_sat[i] = np.mean(x_train_arr[i,:,:,1]) # getting the mean of the hue channel
    mean_of_train_val[i] = np.mean(x_train_arr[i,:,:,2]) # getting the mean of the hue channel


mean_of_train = np.concatenate((mean_of_train_hue, mean_of_train_sat, mean_of_train_val), axis=1)
mean_of_val = np.concatenate((mean_of_val_hue, mean_of_val_sat, mean_of_val_val), axis=1)
# save to csv file
np.savetxt('mean_test_hsv.csv', mean_of_train, delimiter=',')

# Test reading
load_hsv_train = np.loadtxt('mean_test_hsv.csv', dtype=np.float64, delimiter=',')

del y_train
gc.collect()

#%% Loading the images 
x_tr_load = []
dataset_dir = os.path.join(project_dir, 'preprocessing_test') 
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    if case.endswith(".jpg") and not case.startswith("."): # look here for muliclass labeling
        # if (aux%100==0):
        x_tr_load.append(cv2.imread(os.path.join(dataset_dir, case), cv2.IMREAD_GRAYSCALE))
        
x_load_array = np.array(x_tr_load,dtype = np.uint8)
        
#%% Fine Segmentation
# from skimage.morphology import area_opening
# x_l_mask = np.array(x_load_array > 0, dtype = np.uint8)
# x_up_mask = np.array(x_load_array < 200 , dtype = np.uint8)
# x_mask = np.multiply(x_l_mask, x_up_mask)

# x_mask_open = area_opening(np.uint8(x_mask))

# # for i in range(0,x_val_arr.shape[0]):

# x_load_array = np.multiply(x_mask, x_load_array)

                




