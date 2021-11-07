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

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 

# Generator Parameters
img_size = (450, 600,3) # RGB imges!
batch_size = 32

# Sparse implementation for dev speed --> read every 10th image
aux = 0

# Getting paths to images
y_train = []

for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."): # look here for muliclass labeling
                y_train.append(case)

# Sparse implementation for dev speed
aux = 0

# Getting paths to images
y_val = []

for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                y_val.append(case)
                           
#%% Loading of data

# Test reading
load_hsv_train = np.loadtxt('mean_hsv_train.csv', dtype=np.float64, delimiter=',')[0:1200]
load_hsv_val = np.loadtxt('mean_hsv_val.csv', dtype=np.float64, delimiter=',')[0:1200]

x_tr_load = []
dataset_dir = os.path.join(project_dir, 'preprocessing') 
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    if image.endswith(".jpg") and not image.startswith("."): # look here for muliclass labeling
        # if (aux%100==0):
        x_tr_load.append(cv2.imread(os.path.join(dataset_dir, 'train', case), cv2.IMREAD_GRAYSCALE))
        
x_load_array = np.array(x_tr_load,dtype = np.uint8)

x_val_load = []
dataset_dir = os.path.join(project_dir, 'preprocessing') 
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    if image.endswith(".jpg") and not image.startswith("."): # look here for muliclass labeling
        # if (aux%100==0):
        x_tr_load.append(cv2.imread(os.path.join(dataset_dir, 'train', case), cv2.IMREAD_GRAYSCALE))
        
x_load_array = np.array(x_tr_load,dtype = np.uint8)
        
        
#%% Fine Segmentation
# from skimage.morphology import area_opening
# x_l_mask = np.array(x_load_array > 0, dtype = np.uint8)
# x_up_mask = np.array(x_load_array < 200 , dtype = np.uint8)
# x_mask = np.multiply(x_l_mask, x_up_mask)

# x_mask_open = area_opening(np.uint8(x_mask))

# # for i in range(0,x_val_arr.shape[0]):

# x_load_array = np.multiply(x_mask, x_load_array)

                
#%% Feature Extraction

mean_of_train = np.zeros(x_train_arr.shape[0])
mean_of_val = np.zeros(x_val_arr.shape[0])
    
### Color -> Hue ###


# LBP

# del x_train_arr
# del x_val_arr
# gc.collect()

# #%% Inputs

X_train = mean_of_train[np.newaxis].T
Y_train = np.where(np.array(Y_train) == 'les',1,0)


# X_val = feature_vector_val[np.newaxis].T
# Y_val = np.where(np.array(Y_val) == 'les',1,0)

# del feature_vector_train
# del feature_vector_val
# gc.collect()

#%%%%%%%%%%%%%%%%%%%%%%% Feature Selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# #% UNIVARIATE SELECTION
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# select_feature = SelectKBest().fit(X_train, labels_train)

# X_train = select_feature.transform(X_train)
# X_test = select_feature.transform(X_val)

# #%% RANDOM FOREST ELIMINATION
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier

# clf_rf_3 = RandomForestClassifier()      
# rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
# rfe = rfe.fit(X_train, labels_train)

# # breast_data_pd_data.columns[rfe.support_] # checking what we're using

# X_train = X_train.T[rfe.support_].T
# X_val = X_val.T[rfe.support_].T

# #%% RECURSIVE FEATURE ELIMINATION WITH CROSS VALIDATION

# from sklearn.feature_selection import RFECV

# # The "accuracy" scoring is proportional to the number of correct classifications
# clf_rf_4 = RandomForestClassifier() 
# rfecv = RFECV(estimator=clf_rf_4, step=10, cv=10, scoring='accuracy')   #5-fold cross-validation
# rfecv = rfecv.fit(X_train, labels_train)

# X_train = rfecv.transform(X_train)
# X_test = rfecv.transform(X_test)

# #%% LINEAR SCV 
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel

# lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(X_train, labels_train)
# model = SelectFromModel(lsvc, prefit=True)

# X_train = model.transform(X_train)
# X_test = model.transform(X_val)

#%%%%%%%%%%%%%%%%%%% CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,Y_train)
    
# # Predictions
# print(neigh.predict([[1]])) # hard classification prediction
# print(neigh.predict_proba([[0.9]])) # confidence score prediction



