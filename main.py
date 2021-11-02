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

# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'dataset') 

# Generator Parameters
img_size = (450, 600,3) # RGB imges!
batch_size = 32

# Getting paths to images
y_train = []
x_train = []
for case in os.listdir(os.path.join(dataset_dir, 'train')):
    for image in os.listdir(os.path.join(dataset_dir, 'train', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                pseudo_x = cv2.imread(os.path.join(dataset_dir, 'train', case, image))
                x_train.append(pseudo_x)
                y_train.append(case)
    
# Getting paths to images
y_val = []
x_val = []
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                pseudo_x = cv2.imread(os.path.join(dataset_dir, 'val', case, image))
                x_val.append(pseudo_x)
                y_val.append(case)

del pseudo_x              
gc.collect()
    
#%% Preprocessing & Feature Extraction
### RBG ###
from skimage.color import rgb2hsv

# List to array
x_train_arr = np.array(x_train, dtype = np.uint8)
x_val_arr = np.array(x_val, dtype = np.uint8)

# del x_train
# del x_val
# gc.collect()

# SIFT


sift = cv2.SIFT()
kp, des = sift.detectAndCompute(x_train_arr[100])

### Color Space Transformation: RGB --> HSV ###
Y_val = []
Y_train = []

for i in range(0,x_train_arr.shape[0]):
    # x_train_arr[i] = rgb2hsv(x_train_arr[i])
    Y_train.append(y_train[i])
    
for i in range(0,x_val_arr.shape[0]):
    # x_val_arr[i] = rgb2hsv(x_val_arr[i])
    Y_val.append(y_val[i])

del y_train
del y_val
gc.collect()

#%% Feature Extraction

mean_of_train = np.zeros(x_train_arr.shape[0])
mean_of_val = np.zeros(x_val_arr.shape[0])

# Mean of image
for i in range(1,x_train_arr.shape[0]):
    mean_of_train[i] = np.mean(x_train_arr[i,:,:,2])
    
for i in range(1,x_val_arr.shape[0]):
    mean_of_val[i] = np.mean(x_val_arr[i,:,:,2])
    


# LBP

# del x_train_arr
# del x_val_arr
# gc.collect()

# #%% Inputs

# X_train = feature_vector_train[np.newaxis].T
# Y_train = np.where(np.array(Y_train) == 'les',1,0)


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

# neigh = KNeighborsClassifier(n_neighbors=2)
# neigh.fit(X_train,labels_train)
    
# # Predictions
# print(neigh.predict([[1]])) # hard classification prediction
# print(neigh.predict_proba([[0.9]])) # confidence score prediction



