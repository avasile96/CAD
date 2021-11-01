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
                pseudo_x = io.imread(os.path.join(dataset_dir, 'train', case, image), as_gray = False)
                x_train.append(pseudo_x)
                y_train.append(case)
    
# Getting paths to images
y_val = []
x_val = []
for case in os.listdir(os.path.join(dataset_dir, 'val')):
    for image in os.listdir(os.path.join(dataset_dir, 'val', case)):
            if image.endswith(".jpg") and not image.startswith("."):
                pseudo_x = io.imread(os.path.join(dataset_dir, 'val', case, image), as_gray = False)
                x_val.append(pseudo_x)
                y_val.append(case)

pseudo_x = None                
gc.collect()
    
#%% Generator
class SkinImageDatabase(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, img_label):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.img_label = img_label

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_labels = self.img_label[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = io.imread(path, as_gray = False)
            x[j] = img

        y = np.array((self.batch_size,) + tuple(self.img_label), dtype="str")
        for j, path in enumerate(batch_target_img_labels):
            y[j] = path
            
        return x, y

#%% Preprocessing
from skimage.color import rgb2hsv

# List to array
x_train_arr = np.array(x_train)
x_val_arr = np.array(x_val)

x_train = None
x_val = None
gc.collect()

# Color Space Transformation
m = 0
x_train_hsv = np.zeros([60,450,600,3])
x_val_hsv = np.zeros([60,450,600,3])

# Following routine is only for dev, modify for full implementation
for i in range(1,62,10):
    if i!=1:
        x_train_hsv[m:i-1,:,:,:] = rgb2hsv(x_train_arr[m:i-1,:,:,:])
        x_val_hsv[m:i-1,:,:,:] = rgb2hsv(x_val_arr[m:i-1,:,:,:])

        m = i-1
        
    # every 100 images, we collect the garbage (RAM saving)
    if i % 100 == 0:
        gc.collect()

#%% Feature Extraction

feature_vector_train = np.zeros(x_train_hsv.shape[0])
feature_vector_val = np.zeros(x_val_hsv.shape[0])

# Mean of image
for i in range(1,x_hsv.shape[0]):
    feature_vector_train[i] = np.mean(x_hsv[i,:,:,2])
    feature_vector_val = np.zeros(x_val_hsv.shape[0])
    
# SIFT

# LBP

#%% Inputs

x_train = feature_vector[np.newaxis].T
y_train = np.array(img_label_train_train,dtype='U')
np.where(y_train == 'les',1,0)

x_val = 
y_val = 

#%% Feature Selection

#% UNIVARIATE SELECTION
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

select_feature = SelectKBest(chi2, k=20).fit(x_train, y_train)

x_train = select_feature.transform(x_train)
X_test = select_feature.transform(x_val)

#%% RANDOM FOREST ELIMINATION
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

clf_rf_3 = RandomForestClassifier()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)

# breast_data_pd_data.columns[rfe.support_] # checking what we're using

x_train = x_train.T[rfe.support_].T
X_test = X_test.T[rfe.support_].T

#%% RECURSIVE FEATURE ELIMINATION WITH CROSS VALIDATION

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=10, cv=10, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

x_train = rfecv.transform(x_train)
X_test = rfecv.transform(X_test)

#%% LINEAR SCV 
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(x_train, y_train)
model = SelectFromModel(lsvc, prefit=True)

x_train = model.transform(x_train)
X_test = model.transform(X_test)



#%% Classifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x_train,y_train[0:60])
    
# Predictions
print(neigh.predict([[1]])) # hard classification prediction
print(neigh.predict_proba([[0.9]])) # confidence score prediction



